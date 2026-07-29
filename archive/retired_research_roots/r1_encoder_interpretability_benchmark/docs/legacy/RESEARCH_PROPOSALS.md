# Anthropic Interpretability Methods x Biological Language Models: Research Proposals

**Date**: 2026-03-01

---

## Background Summary

### Core Anthropic Interpretability Toolbox

| Method | Core idea | Technical maturity |
|------|---------|-----------|
| **Sparse Autoencoders (SAEs)** | Decompose polysemantic neural activations into monosemantic features | High (open-source implementations already exist) |
| **Cross-Layer Transcoders (CLTs)** | Trace how features are computed from input features to output features across layers | Medium (released in 2025) |
| **Attribution Graphs / Circuit Tracing** | Build a full computation graph: the causal chain from input -> intermediate features -> output | Medium |
| **Feature Steering** | Control model behavior by amplifying or suppressing specific features | High |
| **Activation Patching** | Measure causal contribution by replacing activations in specific components | High |

### Current State of Interpretability in Biological Language Models

| Domain | SAE analysis | Circuit tracing | Feature Steering | Cross-modal interpretation |
|------|---------|---------|-----------------|-----------|
| Protein (ESM-2) | **3 papers already** (InterPLM, InterProt, Gujral) | Only preliminary transcoder attempts | Unexplored | Unexplored |
| DNA (Evo 2) | **1 paper already** (Goodfire) | Unexplored | Preliminary attempts | N/A |
| Molecules/drugs | **Completely open** | Completely open | Completely open | Completely open |
| RNA | **Completely open** | Completely open | Completely open | N/A |
| Text -> biomolecule generation | **Completely open** | Completely open | Completely open | **Completely open** |

---

## Proposal 1: Interpretability Analysis of Text-Conditioned Biomolecular Generation Models ★★★★★

### Research Question
**When a text-conditioned generation model (such as MolT5, BioT5, ProteinDT, or Pinal) converts natural-language descriptions into molecular or protein structures, which semantic features in the text drive which parts of the generated structure?**

### Why This Direction Is the Most Worth Pursuing

1. **A completely open problem**: As of March 2026, no work has studied the internal mechanisms of text -> biomolecule generation.
2. **Dual scientific significance**:
   - **From the AI side**: Reveal the "translation circuits" in cross-modal generation, showing how language concepts are mapped to structural features.
   - **From the biology side**: Test whether the model learns internal representations similar to pharmacophores, which could reveal new structure-activity relationships.
3. **Practical value**: Understanding the generation mechanism could improve prompt design, debug generation failures, and help ensure generation safety.
4. **A direct analogue of Anthropic's work**: Similar to the multi-step reasoning and planning circuits found in "On the Biology of a Large Language Model", but applied to biomolecular generation.

### Concrete Plan

#### Phase 1: Model Selection and Baseline Setup (2-3 weeks)
- **Primary model**: MolT5 (open T5 architecture, bidirectional text <-> molecule translation, active community)
- **Alternatives/extensions**: BioT5+ (multitask), 3D-MolT5 (3D-aware)
- **Baseline tasks**: text -> SMILES generation, molecule captioning
- **Evaluation datasets**: ChEBI-20 (text-molecule pairs), PubChem text descriptions

#### Phase 2: SAE Feature Extraction and Cross-Modal Analysis (4-6 weeks)
- Train SAEs on both encoder and decoder layers of MolT5
- **Key innovation**: Identify "bridging features" that are highly activated for both textual semantic concepts and molecular structural features
- Categorize SAE features into:
  - Pure text features (such as "inhibitor", "aromatic", "binds to")
  - Pure molecular features (such as benzene rings, hydroxyl groups, specific scaffolds)
  - Bridging features (jointly encoding text concepts and molecular structure)
- Validation: Align molecule-side SAE features with chemical ontologies (functional groups, Murcko scaffolds, ADMET properties)

#### Phase 3: Causal Attribution Analysis (3-4 weeks)
- **Attribution Patching**: For each generated SMILES token, trace which text-token features in which encoder layers contribute most
- **Feature Ablation**: Suppress specific text features and observe how the generated molecule changes
  - Example: suppress features related to "aromatic" -> does the generated molecule lose aromatic rings?
  - Example: suppress features related to "inhibitor" -> do the activity-related structures change?
- Build a "conditional dependency graph": text features -> molecular substructures

#### Phase 4: Cross-Modal Circuit Tracing (3-4 weeks)
- Apply Anthropic's circuit tracing methodology to trace the full generation path:
  - "This molecule inhibits kinase activity" -> [text features] -> [bridging features] -> [molecular features] -> concrete SMILES tokens
- Look for phenomena similar to those found in "On the Biology of a Large Language Model":
  - **Multi-step planning**: Does the model first plan the molecular scaffold and then fill in details?
  - **Parallel strategies**: Does it consider multiple candidate structures simultaneously?
  - **Internal pharmacophores**: Does the model develop pharmacophore-like internal representations?

### Expected Discoveries and Publication Goals
- **Core discovery**: A text -> molecule "translation dictionary" showing which language concepts map to which chemical structures
- **Methodological contribution**: The first mechanistic interpretability framework for a cross-modal biological generation model
- **Target journals/conferences**: Nature Machine Intelligence / ICML / NeurIPS (similar venue tier to InterProt)

### Feasibility Assessment: ★★★★☆
- MolT5 is an open T5 architecture, and SAE training tools are mature
- Main challenge: implementing cross-modal attribution will require custom engineering
- Compute requirement: moderate (MolT5 is mid-sized)

---

## Proposal 2: Circuit Tracing in Protein Language Models: The Computational Mechanism from Sequence to Structure ★★★★★

### Research Question
**How does ESM-2 compute implicit three-dimensional structural information from one-dimensional protein sequences? In which layers does this "sequence -> structure" computation happen, and what are the intermediate representations?**

### Why It Is Worth Doing

1. **Extremely high scientific significance**: Protein folding is a core problem in biology. Understanding how a PLM "solves" it would advance both AI and protein science.
2. **A natural extension of prior work**: InterPLM and InterProt have already found SAE features in ESM-2 corresponding to structural motifs, but they did not trace the computational pathway.
3. **Gujral et al. already demonstrated transcoder feasibility**: They successfully trained transcoders on ESM-2, paving the way for circuit tracing.
4. **The best biological testbed for Anthropic's methodology**: The methods from "On the Biology of a Large Language Model" can be transferred almost directly.

### Concrete Plan

#### Phase 1: Extend Existing Work (2-3 weeks)
- Reproduce the transcoder training from Gujral et al. (ESM-2-650M)
- Extend to Cross-Layer Transcoders (CLTs) for feature tracing across layers
- Choose test cases: protein families with known structures (such as TIM barrels and alpha-helical bundles)

#### Phase 2: Structure-Computing Circuit Tracing (4-6 weeks)
- **Core experiment**: When ESM-2 processes a protein sequence, trace the full computational graph from sequence input to implicit structural representation
- Focus on three classes of circuits:
  1. **Local structure circuits**: computation from sequence motifs -> secondary structure (alpha-helices, beta-sheets)
  2. **Long-range contact circuits**: how distant residues establish connections through intermediate features
  3. **Folding hierarchy circuits**: how secondary-structure features compose into supersecondary structures and domains
- **Causal validation**: Use activation patching to test whether the discovered circuits are causally relevant

#### Phase 3: Compare with Protein Science Knowledge (2-3 weeks)
- Compare the discovered computational circuits with known protein-folding rules:
  - Has the model learned a concept similar to a folding nucleus?
  - Are there circuits analogous to hydrophobic-core-driven folding?
  - Does long-range contact computation involve coevolutionary information?

### Expected Discoveries
- The internal "folding algorithm" of ESM-2 may differ fundamentally from physical folding processes, similar to Anthropic's finding that Claude's addition algorithm differs from human algorithms
- It may reveal new sequence-structure relationship rules

### Feasibility Assessment: ★★★★☆
- Technically, this maps closely to Anthropic's work on Claude
- Main challenges: ESM-2-650M requires substantial compute; CLT training needs adaptation for non-language models
- A proof of concept can begin with ESM-2-8M

---

## Proposal 3: SAE Analysis of Molecular Language Models: Discovering the Drug Model's "Internal Pharmacophores" ★★★★☆

### Research Question
**Do molecular language models (ChemBERTa, Uni-Mol) learn internal representations similar to pharmacophores? How do these representations compare with known chemical knowledge?**

### Why It Is Worth Doing

1. **Completely open**: As of March 2026, no SAE or mechanistic interpretability work has been applied to molecular language models
2. **A central problem in drug discovery**: Understanding how the model represents chemical structure directly affects the reliability of drug design
3. **Regulatory pressure**: The FDA's 2025 draft guidance requires interpretability for AI-based drug discovery
4. **Analogy to protein-model success**: InterPLM found features corresponding to biological concepts in ESM-2, and the same approach is highly likely to work for molecular models

### Concrete Plan

#### Phase 1: SAE Training and Basic Analysis (3-4 weeks)
- **Model**: ChemBERTa (77M parameters, SMILES pretraining, RoBERTa architecture) or Molformer
- Train SAEs across layers (expansion factor 8-16x)
- Systematically align SAE features with chemical ontologies:
  - Functional groups
  - Murcko scaffolds
  - ECFP fingerprint bits
  - Pharmacophore features (hydrogen bond donor/acceptor, hydrophobic, aromatic, etc.)
  - ADMET properties

#### Phase 2: Pharmacophore Feature Analysis (3-4 weeks)
- Core question: do SAE features correspond to 3D pharmacophores?
  - Traditional pharmacophores are 3D concepts, but molecular models learn from 1D SMILES
  - Analogy: ESM-2 learns 3D structural information from 1D sequence
- Analyze whether SAE features can predict:
  - Target specificity
  - Toxicity mechanisms
  - Metabolic stability
  - Drug-drug interactions

#### Phase 3: Feature Steering Experiments (2-3 weeks)
- If using a generative molecular model (such as a GPT-based model):
  - Activate "thermal stability" features -> generate more stable molecules?
  - Suppress "hepatotoxicity" features -> avoid liver toxicity?
- Validation: use ADMET prediction tools to test the steering results

### Feasibility Assessment: ★★★★★
- ChemBERTa is a mature open-source model with a simple architecture
- SAE training pipelines already have mature tooling (such as SAELens)
- Chemical ontologies are mature and validation methods are rich
- Low compute requirement (small model)
- **The easiest direction to execute successfully**

---

## Proposal 4: Feature-Steered Protein Engineering ★★★★☆

### Research Question
**Can we achieve directed modification of protein properties by manipulating SAE features in ESM-2 or ESM-3? For example, could activating "thermal stability" features help design thermostable proteins?**

### Why It Is Worth Doing

1. **Direct practical value**: If successful, this would provide a fundamentally new protein engineering strategy
2. **A direct analogue of Anthropic's "Golden Gate Claude"**: Feature steering has already been shown to work in language models
3. **Goodfire's early success on Evo2**: DNA-model steering has precedent, and proteins are a natural extension
4. **Strong verifiability**: Protein stability, activity, solubility, and related properties have mature experimental and computational validation methods

### Concrete Plan

#### Phase 1: Identify Functional SAE Features (2-3 weeks)
- Use SAE features from InterPLM/InterProt (or retrain them)
- Systematically align features with protein engineering targets:
  - Thermal stability (Tm)
  - Solubility
  - Catalytic activity
  - Specificity
  - Expression level

#### Phase 2: Feature Steering Experiments (3-4 weeks)
- Perform feature manipulation on protein sequences generated by ESM-2 or ESM-3:
  - Activate thermal-stability features -> generate thermostable proteins
  - Suppress aggregation-related features -> generate highly soluble proteins
  - Combined operations: adjust multiple features simultaneously
- Validate structural plausibility with ESMFold/AlphaFold
- Validate sequence-structure consistency with tools such as ProteinMPNN

#### Phase 3: Compare with Traditional Methods (2-3 weeks)
- Compare against directed evolution and rational design strategies
- Analyze whether feature steering can rediscover known stabilizing mutation strategies
- Test whether it uncovers genuinely novel strategies

### Feasibility Assessment: ★★★☆☆
- Depends on access to ESM-3 (EvolutionaryScale API/model)
- The effect of feature steering in biological models remains uncertain
- Computational validation is feasible, but experimental validation would require collaboration
- **High risk, high reward**

---

## Proposal 5: SAE Interpretability Analysis of RNA Foundation Models ★★★☆☆

### Research Question
**What RNA biological rules do RNA language models (RiNALMo, RNA-FM) learn? Can SAEs reveal internal representations of RNA secondary structure, functional motifs, and regulatory elements?**

### Why It Is Worth Doing

1. **Completely open**: No SAE or mechanistic interpretability work exists here
2. **The rise of RNA therapeutics**: mRNA vaccines, ASOs, and siRNA make understanding RNA structure representations highly important for drug design
3. **A "Goldilocks" scale**: RNA models (<=650M parameters) are very tractable computationally
4. **Unique validation opportunities**: RNA secondary structure can be measured precisely with experiments such as SHAPE and DMS

### Concrete Plan
- Train SAEs on RiNALMo (650M)
- Validate whether features correspond to stem-loops, pseudoknots, riboswitches, miRNA seed regions, and related motifs
- Compare against RNA secondary-structure prediction results

### Feasibility Assessment: ★★★★★
- Small models and low compute requirements
- Rich RNA structural validation data
- But the scientific impact may be lower than the earlier directions (more incremental)

---

## Comprehensive Comparison

| Direction | Novelty | Scientific impact | Feasibility | Compute requirement | Publication potential | Overall recommendation |
|------|--------|---------|--------|---------|---------|---------|
| **P1: Text -> molecule generation interpretability** | ★★★★★ | ★★★★★ | ★★★★☆ | Medium | Nature MI / ICML | **Top recommendation** |
| **P2: ESM-2 circuit tracing** | ★★★★☆ | ★★★★★ | ★★★★☆ | High | Nature Methods / ICML | **Second recommendation** |
| **P3: SAE analysis of molecular models** | ★★★★★ | ★★★★☆ | ★★★★★ | Low | NeurIPS / JCIM | **Easiest to execute** |
| **P4: Protein Feature Steering** | ★★★★☆ | ★★★★☆ | ★★★☆☆ | Medium-high | Nature Biotech / ICLR | High risk, high reward |
| **P5: RNA-model SAEs** | ★★★★★ | ★★★☆☆ | ★★★★★ | Low | Bioinformatics / NAR | Safe but incremental |

---

## Implementation Recommendations

### If Resources Are Abundant (multi-GPU, 3-6 months):
-> **P1 (text -> molecule generation interpretability)** as the main line, with P3 (SAE analysis of molecular models) as foundational support

**Reason**: P1 opens an entirely new research space, while P3 can serve as a foundational module for P1: first understand how molecular models represent chemical knowledge, then analyze how cross-modal generation uses that knowledge.

### If Resources Are Limited (single GPU, 1-3 months):
-> **P3 (SAE analysis of molecular models)** for faster results, while also laying groundwork for P1

**Reason**: ChemBERTa is small, SAE training is mature, and chemical validation tooling is rich, so it should be possible to produce solid results within 1-2 months.

### If Maximizing Scientific Impact Is the Goal:
-> **P2 (ESM-2 circuit tracing)**

**Reason**: Revealing the computational mechanism of protein folding would be a major cross-disciplinary discovery, but it requires more compute and greater technical difficulty.
