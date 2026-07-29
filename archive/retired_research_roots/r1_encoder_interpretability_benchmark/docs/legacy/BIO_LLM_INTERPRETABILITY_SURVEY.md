# Comprehensive Survey: AI Interpretability Methods for Biological Language Models

**Date**: March 2026
**Scope**: Protein, molecule/drug, DNA/RNA, single-cell, and multi-modal biological language models; interpretability techniques applied to each; text-conditioned generation models; key open problems and research gaps.

---

## Table of Contents

1. [Current Biological Language Models](#1-current-biological-language-models)
2. [Existing Interpretability Work on Bio-Models](#2-existing-interpretability-work-on-bio-models)
3. [Key Open Problems in Bio-LLM Interpretability](#3-key-open-problems-in-bio-llm-interpretability)
4. [Text-Conditioned Molecular and Protein Generation Models](#4-text-conditioned-molecular-and-protein-generation-models)
5. [High-Impact Research Gaps](#5-high-impact-research-gaps)
6. [References](#6-references)

---

## 1. Current Biological Language Models

### 1.1 Protein Language Models

| Model | Organization | Parameters | Architecture | Key Features |
|-------|-------------|-----------|--------------|--------------|
| **ESM-2** | Meta/FAIR | 8M to 15B | Masked LM (BERT-style) | Evolutionary scale modeling; contact prediction; structure prediction |
| **ESM-3** | EvolutionaryScale | 98B | Multitrack transformer | Unified sequence + structure + function; geometric attention; generative |
| **ESM-C** | EvolutionaryScale | 300M/600M | Encoder | Compact, efficient protein encoder |
| **ProtTrans** (ProtBERT, ProtT5, etc.) | Rostlab | Up to 11B | BERT/T5 variants | Trained on UniRef/BFD; multilingual protein "languages" |
| **AlphaFold 2/3** | DeepMind | -- | Evoformer + structure module | Structure prediction from MSA + templates; AF3 adds ligands, nucleic acids |
| **ProGen / ProGen2** | Salesforce | Up to 6.4B | Autoregressive | Protein sequence generation conditioned on properties |
| **METL** | -- | Varies | Transformer | Biophysics-based PLM pretrained on simulation data (Nature Methods, 2025) |
| **ProTrek** | -- | -- | Trimodal contrastive | Unifies sequence, structure, and natural language (Nature Biotechnology) |

**What ESM-family models learn**: ESM-2 captures coevolutionary statistics analogous to Markov Random Fields. The "categorical Jacobian" analysis shows ESM-2 stores pairwise residue dependencies conditioned on sequence motifs and residue separation. Larger models implicitly learn contact maps, secondary structure, and functional site locations from sequence alone.

### 1.2 Molecular / Drug Language Models

| Model | Type | Architecture | Key Features |
|-------|------|-------------|--------------|
| **MolBERT** | Small molecule | BERT on SMILES | Pretrained molecular representations; fingerprint/property prediction |
| **ChemBERTa** | Small molecule | RoBERTa on SMILES | Pretrained on 77M SMILES; molecular property prediction |
| **ChemLM** | Small molecule | Domain-adapted LM | Two-stage training with augmented SMILES (Nature Comm Chem, 2025) |
| **GPT-MolBERTa** | Small molecule | GPT + MolBERTa | Attention-based interpretability for molecular descriptions |
| **Uni-Mol** | 3D molecule | 3D transformer | 209M conformations; separate molecular and pocket models |
| **Uni-Mol2** | 3D molecule | Dual-track transformer | 1.1B params; 800M conformations; scaling laws demonstrated (NeurIPS 2024) |
| **Uni-Mol3** | Reaction modeling | Hierarchical pipeline | Multi-molecular tokenizer; 3D-aware molecular language (2025) |
| **Molformer** | Small molecule | Transformer | Efficient molecular representations |

### 1.3 DNA/RNA Language Models

| Model | Modality | Parameters | Key Features |
|-------|---------|-----------|--------------|
| **DNABERT** | DNA | ~110M | K-mer tokenization; DNABERT-viz for interpretability |
| **DNABERT-2** | DNA | ~117M | BPE tokenization; multi-species genomes; improved efficiency |
| **Nucleotide Transformer (NT)** | DNA | 50M to 2.5B | Multi-species; NT-v2 with 12kb context |
| **Evo** | DNA | 7B | StripedHyena architecture; 131kb context; single-nucleotide resolution |
| **Evo 2** | DNA | 7B/40B | 9.3T base pairs; 1M token context; all domains of life |
| **HyenaDNA** | DNA | -- | Long-context genomic model using Hyena operator |
| **RNA-FM** | RNA | ~100M | 23.7M non-coding RNA; interpretable evolutionary embeddings |
| **RNABERT** | RNA | -- | Secondary structure prediction; family classification |
| **RiNALMo** | RNA | Up to 650M | Generalizes to unseen RNA families; clean family clustering |
| **PlantRNA-FM** | RNA (plants) | -- | Functional RNA motif identification; interpretable framework (F1=0.974) |
| **ERNIE-RNA** | RNA | -- | Strong family separation in embedding space |

### 1.4 Single-Cell Foundation Models

| Model | Architecture | Training Data | Key Features |
|-------|-------------|--------------|--------------|
| **scGPT** | GPT-style autoregressive | 33M+ cells | Gene expression modeling; cell type annotation; perturbation prediction |
| **Geneformer** | BERT-style masked prediction | 30M+ cells | Rank-ordered gene tokens; transfer learning across tissues |
| **scBERT** | BERT encoder | -- | Cell type annotation from scRNA-seq |
| **scFoundation** | Large-scale FM | 50M+ cells | Broad multi-task single-cell model |
| **scPRINT** | -- | -- | Single-cell pretrained model |
| **UCE** | Universal cell embeddings | -- | Cross-species cell representations |
| **Nicheformer** | -- | -- | Spatial and single-cell integration |

### 1.5 Multi-Modal Biological Models

| Model | Modalities | Key Features |
|-------|-----------|--------------|
| **BioMedGPT** | Molecules + proteins + text | 10B params; Q&A on molecules/proteins; BioMedGPT-R1 adds reasoning (2025) |
| **GIT-Mol** | Graphs + images + text (SMILES + captions) | GIT-Former for unified latent space (Comp Bio Med, 2024) |
| **MolLM** | 2D/3D molecular + text | Unified biomedical text with molecular representations |
| **BioT5/BioT5+** | Molecules + text | T5-based; cross-modal biology + chemistry (EMNLP 2023) |
| **3D-MoLM** | 3D molecule + text | 3D molecule-text interpretation (ICLR 2024) |
| **MolPrompt** | Molecular graphs + text | Knowledge-enhanced contrastive pretraining (Bioinformatics, 2025) |
| **OpenBioMed / PharMolixFM** | Molecules + antibodies + proteins | Atom-level unified modeling (v2 released March 2025) |
| **Mol-LLM** | Text + molecular graphs | Generalist: translation, prediction, generation in one model (2025) |
| **Omni-Mol** | Any-to-any molecular modalities | Multitask; NeurIPS 2025 |

---

## 2. Existing Interpretability Work on Bio-Models

### 2.1 Sparse Autoencoders (SAEs) on Protein Language Models

This is the most active and well-developed area at the intersection of mechanistic interpretability and biology. Three landmark papers appeared in 2024-2025:

#### InterPLM (Simon & Zou, 2024 preprint; Nature Methods, 2025)
- **Approach**: Trained SAEs with L1 regularization on ESM-2-8M embeddings (hidden dim = 10,420).
- **Key findings**:
  - Identified up to 2,548 human-interpretable latent features per layer, correlating with 143 known biological concepts (binding sites, structural motifs, functional domains).
  - Individual ESM-2 neurons showed alignment with only ~15 concepts per layer (~46 neurons), confirming that PLMs represent most biological concepts in **superposition**.
  - SAE features discovered annotation gaps: features identified a Nudix box motif in a protein that was missing from Swiss-Prot but confirmed in InterPro.
- **Significance**: First demonstration that the SAE interpretability framework from NLP transfers productively to protein biology.

#### InterProt (Adams, Bai, Lee, Yu & AlQuraishi, 2025 -- ICML Spotlight)
- **Approach**: Trained TopK SAEs on the residual stream of ESM-2-650M. Developed the InterProt visualization tool for exploring latent activations on protein sequences and structures.
- **Key findings**:
  - PLMs use a combination of **generic features** (shared across protein families) and **family-specific features**.
  - Middle layers contain the most family-specific features.
  - Linear probes on SAE features identify known sequence determinants of thermostability, subcellular localization, and other properties.
  - SAE latents consistently rated as more interpretable than raw ESM neurons in human evaluation.
  - Predictive features without known functional associations can generate new biological hypotheses.
- **Significance**: Scaled SAE interpretability to a production-scale PLM; introduced hypothesis generation from unexplained features.

#### Gujral, Bafna, Alm & Berger (PNAS, 2025)
- **Approach**: Used both SAEs and **transcoders** (a variant that traces how upstream features contribute to downstream activations) on ESM-2. Completely unsupervised feature extraction at both protein-level and amino-acid-level.
- **Key findings**:
  - Many sparse features tightly associated with Gene Ontology (GO) terms across all levels of the hierarchy.
  - Used Anthropic's Claude LLM to **automate interpretation** of sparse features -- found many correspond to specific protein families (NAD Kinase, IUNH, PTH family, etc.).
  - Transcoders enable circuit-level analysis across layers, a step beyond static feature identification.
- **Significance**: Introduced transcoders to biology; demonstrated automated interpretability pipelines.

#### Anthropic's Circuits Updates -- July 2025 Review
- Reviewed all five major papers applying SAEs to biological models (InterPLM, InterProt, Markov Bio, Reticular, Evo 2).
- Key framing: "The superposition hypothesis holds even more strongly here -- individual neurons in protein models entangle dozens of biological concepts that SAEs successfully disentangle."
- Emphasized dual value: (1) validating that models genuinely learned biology, and (2) using interpretable features to discover new biology (annotation gaps, unknown mechanisms).

### 2.2 SAEs on DNA Foundation Models (Evo 2)

#### Goodfire x Arc Institute Collaboration (2025)
- **Approach**: Trained BatchTopK SAEs on layer 26 of Evo 2 (7B/40B parameter DNA foundation model).
- **Key biological features discovered**:
  - Transcription factor binding sites
  - Exon-intron boundaries
  - Protein structural motifs (alpha-helices, beta-sheets) -- from DNA sequence alone
  - Mobile genetic elements, prophages, CRISPR-associated sequences
  - Canonical gene structures (CDS, UTRs, exons) and RNA stem-loops
- **Validation**: Large-scale alignment analysis between biological concepts and SAE features using domain-F1 scores.
- **Remarkable finding**: Features that activated on protein secondary structures (alpha-helices, beta-sheets) could be mapped onto AlphaFold 3 structure predictions -- the model learned DNA-to-protein-structure correspondence purely from genomic sequences.
- **Steering experiments**: Early signs that Evo 2 can be steered to engineer protein structures by manipulating features, though steering a "nucleotide-only" model is considerably more complex than steering a language model.

### 2.3 Attention-Based Interpretability of Protein LMs

#### High Attention Sites (PLOS Computational Biology, 2025)
- Analyzed ESM attention mechanism to identify "High Attention (HA)" sites.
- HA sites are ubiquitous across human proteins, reliably near functional regions, and robust across protein families.
- Layer-specific attention patterns: Layer 0 shows uniform attention; Layers 10+ show self-attention and localized patterns; Layers 20-32 show fine-grained, residue-specific attention.
- HA sites overlap with active sites, suggesting PLMs implicitly capture functional signals.

#### Knowledge Neurons in ESM (Zhang et al.)
- Identified "knowledge neurons" in ESM after fine-tuning for enzyme classification.
- High density of knowledge neurons found in key vector prediction networks of self-attention modules.
- Knowledge neurons may specialize in capturing different enzyme sequence motifs.

#### Coevolutionary Statistics (PNAS, 2024)
- ESM-2 stores statistics of coevolving residues via the "categorical Jacobian."
- Both MSA-based methods and PLMs ultimately learn to extract coevolutionary information, though through different mechanisms.

### 2.4 Interpretability of Single-Cell Foundation Models

#### Comprehensive Stress-Test (Kendiukhov, 2025)
- 37 analyses, 153 statistical tests, 4 cell types on scGPT and Geneformer.
- **Attention encodes biology but is not uniquely informative**: Layer-specific organization (protein-protein interactions in early layers, transcriptional regulation in late layers), but trivial gene-level baselines outperform attention edges for perturbation prediction (AUROC 0.81-0.88 vs 0.70).
- **Attention-based GRN extraction fails**: Heads most aligned with known regulation are most dispensable for the model's computation.
- **Activation patching** has large non-additivity bias in these models.

#### Spectral Geometry Analysis (Kendiukhov, 2026)
- scGPT organizes genes into a multi-dimensional biological coordinate system: spectral directions encode subcellular localization, PPI networks, and transcriptional regulation.
- 14.4-fold spectral compression (rank decrease) across layers -- substantially more extreme than in language models.
- Residual stream geometry, not attention, is where biologically meaningful representations reside.

#### Topological Structure (2026)
- 141 geometric/topological hypotheses tested across 52 iterations.
- Gene embeddings show non-trivial topology (persistent homology significant in 11/12 layers).
- Models learn genuine geometric structure encoding biological relationships.

### 2.5 Interpretability of Molecular / Drug Models

This area is substantially less developed compared to protein and genomic models:

- **Attention visualization** (BertViz-style): Attention heads in ChemBERTa identify functional groups (hydroxyl, carbonyl, aromatic rings) relevant to chemical properties.
- **SHAP and saliency maps**: Standard XAI methods applied to molecular property prediction models.
- **GNNExplainer**: Applied to graph-based molecular models for substructure identification.
- **No SAE or circuit-level analysis has been published** for molecular language models as of early 2026. This represents a significant gap.

### 2.6 Interpretability of RNA Models

- **RNA-FM**: Authors analyzed learned representations, showing they contain evolutionary information; embeddings can infer evolutionary trends of lncRNAs and SARS-CoV-2 variants.
- **PlantRNA-FM**: Interpretable framework determines critical 5'UTR regions impacting translation.
- **RiNALMo**: Clean family-level clustering in embedding space, suggesting structured representation learning.
- **No SAE-based mechanistic interpretability** has been applied to RNA foundation models yet.

---

## 3. Key Open Problems in Bio-LLM Interpretability

### 3.1 How Do Protein LMs Represent 3D Structure from Sequence Alone?

**What is known**:
- ESM-2 attention patterns correlate with contact maps.
- Coevolutionary statistics (categorical Jacobian) are stored internally.
- SAE features activate on structural motifs (alpha-helices, beta-sheets, loops).

**What remains unknown**:
- The **computational mechanism** by which PLMs convert 1D sequence information into implicit 3D structural representations is not understood at the circuit level.
- How do residue-level features compose into higher-order structural features (domains, folds)?
- Which layers perform the "sequence-to-structure" computation, and what are the intermediate representations?
- How do PLMs distinguish between local structure (secondary structure) and long-range contacts?

**Why it matters**: Understanding this would illuminate both protein folding principles and enable more targeted model engineering for structure prediction.

### 3.2 What Do Attention Patterns in Protein LMs Capture About Biology?

**What is known**:
- HA sites correlate with functional residues and active sites.
- Attention patterns show layer-specific organization.
- Some attention heads specialize in different types of relationships (local vs. long-range).

**What is contested**:
- Whether attention patterns are causally meaningful for model predictions, or merely correlational artifacts.
- The scGPT/Geneformer results (Kendiukhov 2025) suggest that attention encodes biological structure but may not be the primary computational mechanism -- residual stream representations may matter more.

**Key question**: Is attention in PLMs primarily a routing mechanism (as in language models), with the actual biological computation occurring in MLP/FFN layers?

### 3.3 How Do Multi-Modal Models Bridge Between Modalities?

**Current landscape**:
- Models like BioMedGPT, GIT-Mol, ProTrek use various alignment strategies (contrastive learning, cross-attention, shared latent spaces).
- No mechanistic interpretability work has been conducted on how these models internally translate between molecular structure representations and natural language.

**Open questions**:
- What features in the shared latent space correspond to which biological concepts vs. linguistic concepts?
- How does contrastive pretraining (e.g., ProteinCLAP, CLIP-style) shape the geometry of the joint embedding space?
- Are there "bridging features" that activate for both a molecular structural concept and its textual description?

### 3.4 How Do Conditional Generation Models Use Text Features?

**Current landscape**:
- Pinal (16B params) uses a two-stage process: text -> structure -> sequence.
- ProteinDT uses ProteinCLAP alignment + facilitator + decoder.
- MolT5, BioT5, 3D-MolT5 use T5 architectures for molecule-text translation.

**Open questions**:
- In text-to-molecule generation, which text tokens drive which structural decisions?
- How is the mapping from abstract descriptions ("binds to kinase active site") to specific molecular features (hydrogen bond donors at positions X, Y, Z) implemented computationally?
- Do these models develop internal "pharmacophore-like" representations?
- Can we identify circuits responsible for enforcing chemical validity in generated molecules?

---

## 4. Text-Conditioned Molecular and Protein Generation Models

### 4.1 Text-to-Molecule Models

| Model | Architecture | Capability | Key Reference |
|-------|-------------|-----------|---------------|
| **Text2Mol** | Cross-modal retrieval | Retrieves molecules matching text descriptions | Edwards et al. (EMNLP 2021) |
| **MolT5** | T5-based | Bidirectional: molecule captioning + text-to-molecule generation | Edwards et al. (2022) |
| **BioT5 / BioT5+** | T5-based | Joint molecule-text; multitask tuning | Pei et al. (EMNLP 2023) |
| **3D-MolT5** | T5 + 3D tokens | 3D-aware molecule-text translation; ~50% exact match | (2024) |
| **MolXPT** | GPT-style | Wraps molecules with text for generative pretraining | (2023) |
| **Llamole** | Multimodal LLM | Inverse molecular design with retrosynthetic planning | ICLR 2025 |
| **Omni-Mol** | Any-to-any | Multitask molecular model across all modalities | NeurIPS 2025 |
| **PEIT-GEN / PEIT-LLM** | Multi-modal | Text + SMILES + properties; outperforms MolT5/BioT5 on captioning | (2024-2025) |
| **Mol-LLM** | Generalist LLM | Jointly processes text and graphs; translation + prediction + generation | (2025) |
| **ChemLML** | Modular adapters | Blends text and molecules with modular adapters | J Chem Inf Model (2025) |

### 4.2 Text-to-Protein Models

| Model | Architecture | Capability | Key Reference |
|-------|-------------|-----------|---------------|
| **Pinal** | 16B params; two-stage | Text -> structure -> sequence; 1.7B protein-text pairs | bioRxiv (April 2025) |
| **ProteinDT** | ProteinCLAP + facilitator + decoder | Text-guided protein design; 441K text-protein pairs | Nature Machine Intelligence (2025) |
| **ProTrek** | Trimodal contrastive | Sequence + structure + text; 5B protein embeddings | Nature Biotechnology |
| **ESM-3** | Multitrack generative | Prompted with function annotations (text-like); generates sequences/structures | EvolutionaryScale (2024) |
| **ProGen / ProGen2** | Autoregressive | Property-conditioned protein generation | Salesforce |

### 4.3 Drug-Relevant Cross-Modal Models

| Model | Focus | Key Feature |
|-------|-------|------------|
| **DrugCLIP** | Drug-target interaction | Joint molecule-pocket representations via contrastive learning |
| **MolProphecy** | Property prediction | Uses ChatGPT as "virtual chemist" for expert reasoning (2025) |
| **ChemBERTaDDI** | Drug-drug interaction | Molecular embeddings + clinical side effect data |
| **BioMedGPT-Mol** | Molecular understanding | Optical structure + biomedical multimodal reasoning |

### 4.4 Key Technical Approaches in Text-Conditioned Generation

1. **Contrastive alignment** (CLIP-style): ProteinCLAP, DrugCLIP, ProTrek -- learn joint embedding spaces between text and biomolecules.
2. **Two-stage generation**: Text -> intermediate representation (structure/embedding) -> sequence (Pinal, ProteinDT). Constrains the search space through tractable intermediates.
3. **Discrete structure tokens**: Foldseek tokens (Pinal, ProTrek) enable treating 3D structures as sequences amenable to language modeling.
4. **Instruction tuning**: BioT5+, PEIT-LLM, LlaSMol adapt molecular models via instruction-following.
5. **Generalist architectures**: Mol-LLM, Omni-Mol unify translation, prediction, and generation in single frameworks.
6. **Large-scale paired data**: Ranges from 441K (SwissProtCLAP) to 1.7B (Pinal) text-protein pairs.

---

## 5. High-Impact Research Gaps

### Gap 1: SAEs for Molecular / Drug Language Models (HIGH PRIORITY)

**Current state**: No published SAE or mechanistic interpretability work on molecular language models (ChemBERTa, MolBERT, Uni-Mol, Molformer).

**Why it matters**: Drug discovery is a trillion-dollar industry where model failures have real-world consequences. Understanding what molecular LMs learn about chemical structure, reactivity, toxicity, and pharmacological properties could:
- Identify which molecular features drive toxicity predictions (regulatory value -- FDA 2025 draft guidance).
- Reveal whether models learn pharmacophore-like representations.
- Enable targeted model editing for safety-critical applications.
- Discover novel structure-activity relationships encoded in model weights.

**Recommended approach**: Train SAEs on ChemBERTa or Uni-Mol hidden states, then systematically associate features with known chemical ontologies (functional groups, pharmacophores, Murcko scaffolds, ADMET properties).

### Gap 2: Circuit-Level Analysis of Any Biological Model

**Current state**: SAE features have been identified, but no one has traced complete circuits (input -> intermediate features -> output) in any biological model.

**Why it matters**: Feature identification tells us *what* a model represents but not *how* it computes. For protein models, understanding the circuit that converts sequence to structure would be a fundamental contribution to both AI interpretability and protein science.

**Recommended approach**: Apply Anthropic's circuit tracing / attribution graph methods (released May 2025) to ESM-2 for specific tasks (e.g., contact prediction, function annotation). Use transcoders (already applied by Gujral et al.) to trace feature interactions across layers.

### Gap 3: Interpretability of Text-to-Biomolecule Generation

**Current state**: No interpretability work on how text-conditioned models (Pinal, ProteinDT, MolT5, BioT5) translate natural language descriptions into molecular structures.

**Why it matters**: If we cannot understand how these models use text instructions, we cannot:
- Debug failure modes when generation produces incorrect molecules.
- Verify that the model is attending to the right textual features.
- Ensure safe generation (no toxic/dangerous molecules from ambiguous prompts).
- Improve prompt engineering for better generation quality.

**Recommended approach**: Apply attention visualization and feature attribution (integrated gradients, SHAP) to trace which text tokens influence which structural decisions during generation. Train SAEs on the cross-modal attention layers to identify "bridging features."

### Gap 4: Interpretability of RNA Foundation Models

**Current state**: Minimal mechanistic interpretability work. RNA-FM has some representation analysis, but no SAE-based or circuit-level work exists.

**Why it matters**: RNA biology is central to mRNA therapeutics, CRISPR guide design, and understanding non-coding RNA function. RNA models are smaller (up to 650M params) and therefore more tractable for mechanistic analysis.

**Recommended approach**: Apply SAEs to RiNALMo or RNA-FM, focusing on whether models learn RNA secondary structure motifs, tertiary contacts, and regulatory elements. These models present a good "Goldilocks" scale for interpretability work.

### Gap 5: Cross-Model Comparison of Learned Representations

**Current state**: Each biological model family (protein, DNA, RNA, molecule) has been studied in isolation. No systematic comparison of what different model architectures learn about overlapping biological concepts.

**Why it matters**: Proteins are encoded in DNA; RNA structure affects protein expression; drug molecules interact with protein targets. A unified understanding of how different models represent shared biological concepts (e.g., "binding site," "catalytic activity," "structural motif") would:
- Reveal universal vs. modality-specific computational strategies.
- Inform the design of better multi-modal models.
- Enable knowledge transfer between modalities.

**Recommended approach**: Train SAEs on ESM-2, Evo 2, RNA-FM, and ChemBERTa; then use representation alignment techniques (CKA, SVCCA, mutual information) to compare features across models for overlapping biological concepts.

### Gap 6: Interpretability for Model Safety in Therapeutic Applications

**Current state**: Regulatory agencies (FDA, EMA) are mandating transparency for AI in drug development. Current XAI methods (SHAP, attention viz) are insufficient for the depth of understanding needed.

**Why it matters**: As AI models are increasingly used for drug design, protein engineering, and gene therapy design, understanding failure modes becomes safety-critical. Mechanistic interpretability could:
- Identify systematic biases in training data encoded as features.
- Detect when a model is "uncertain" at the feature level (not just output probability).
- Provide auditable explanations for regulatory submission.

### Gap 7: Steering Biological Models via Feature Manipulation

**Current state**: Goodfire's early experiments on steering Evo 2 show promise but are substantially harder than steering language models.

**Why it matters**: If we could reliably steer biological models by activating/suppressing specific features, this would enable:
- Targeted protein engineering (activate "thermostable" features while preserving function).
- Controlled molecule generation (suppress "hepatotoxicity" features during drug design).
- Hypothesis testing (what happens to model predictions when specific biological features are ablated?).

**Recommended approach**: Build on Goodfire's Evo 2 steering work; extend to protein models (ESM-2/3) and molecular models. Develop feature-level control interfaces for biological generation tasks.

---

## 6. References

### SAEs on Protein Language Models
- Simon & Zou. [InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders](https://pubmed.ncbi.nlm.nih.gov/41023434/). Nature Methods, 2025; 22(10):2107-2117.
- Adams, Bai, Lee, Yu & AlQuraishi. [From Mechanistic Interpretability to Mechanistic Biology](https://openreview.net/forum?id=zdOGBRQEbz). ICML 2025 (Spotlight).
- Gujral, Bafna, Alm & Berger. [Sparse autoencoders uncover biologically interpretable features in protein language model representations](https://www.pnas.org/doi/10.1073/pnas.2506316122). PNAS, 2025; 122(34).

### SAEs on DNA Models
- Goodfire & Arc Institute. [Interpreting Evo 2](https://www.goodfire.ai/research/interpreting-evo-2). 2025.
- Arc Institute. [Evo 2: Genome modeling and design across all domains of life](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1). bioRxiv, 2025.

### Protein LM Interpretability
- [Paying attention to attention: High attention sites as indicators of protein family and function](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013424). PLOS Computational Biology, 2025.
- [Identification of Knowledge Neurons in Protein Language Models](https://arxiv.org/html/2312.10770v1). arXiv.
- [Protein language models learn evolutionary statistics of interacting sequence motifs](https://www.pnas.org/doi/10.1073/pnas.2406285121). PNAS, 2024.

### Single-Cell Model Interpretability
- Kendiukhov. [Systematic Evaluation of Single-Cell Foundation Model Interpretability](https://arxiv.org/html/2602.17532v1). arXiv, 2025.
- Kendiukhov. [Multi-Dimensional Spectral Geometry of Biological Knowledge in Single-Cell Transformer Representations](https://arxiv.org/html/2602.22247). arXiv, 2026.
- [What Topological and Geometric Structure Do Biological Foundation Models Learn?](https://arxiv.org/html/2602.22289). arXiv, 2026.

### Text-Conditioned Generation
- [Toward De Novo Protein Design from Natural Language (Pinal)](https://www.biorxiv.org/content/10.1101/2024.08.01.606258v5). bioRxiv, 2025.
- [ProteinDT: A Text-guided Protein Design Framework](https://www.nature.com/articles/s42256-025-01011-z). Nature Machine Intelligence, 2025.
- [ProTrek: A trimodal protein language model](https://www.nature.com/articles/s41587-025-02836-0). Nature Biotechnology.
- [3D-MolT5: Towards Unified 3D Molecule-Text Modeling](https://arxiv.org/html/2406.05797v1). 2024.

### Multi-Modal Models
- [BioMedGPT: Open Multimodal Large Language Model for BioMedicine](https://pubmed.ncbi.nlm.nih.gov/40030352/). 2025.
- [GIT-Mol: A multi-modal large language model for molecular science](https://www.sciencedirect.com/science/article/abs/pii/S0010482524001574). Computers in Biology and Medicine, 2024.
- [Mol-LLM: Multimodal Generalist Molecular LLM](https://arxiv.org/html/2502.02810v2). 2025.

### DNA/RNA Models
- [Nucleotide Transformer](https://www.nature.com/articles/s41592-024-02523-z). Nature Methods, 2024.
- [RiNALMo: general-purpose RNA language models](https://www.nature.com/articles/s41467-025-60872-5). Nature Communications, 2025.
- [An interpretable RNA foundation model for exploring functional RNA motifs in plants](https://pmc.ncbi.nlm.nih.gov/articles/PMC11652376/). 2024.

### Molecular Models
- [Uni-Mol2](https://blogs.deepmodeling.com/Uni-Mol2_18_12_2024/). NeurIPS 2024.
- [ChemLM: Domain adaptable language modeling of chemical compounds](https://www.nature.com/articles/s42004-025-01484-4). Nature Comm Chem, 2025.

### Surveys and Reviews
- [Large Language Models in Bioinformatics: A Survey](https://arxiv.org/html/2503.04490v1). arXiv, 2025.
- [Comprehensive Survey of Multimodal LLMs for Scientific Discovery](https://openreview.net/pdf?id=HSz1Kr5BeC). OpenReview.
- [Single-cell foundation models](https://www.nature.com/articles/s12276-025-01547-5). Experimental & Molecular Medicine, 2025.
- [Explainable AI in Drug Discovery](https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.70049). WIREs Comp Mol Sci, 2025.
- Anthropic. [Circuits Updates - July 2025](https://transformer-circuits.pub/2025/july-update/index.html).
- [A Comprehensive Review of Protein Language Models](https://arxiv.org/html/2502.06881v1). arXiv, 2025.
- [LLMs for Drug Discovery and Development](https://www.cell.com/patterns/fulltext/S2666-3899(25)00194-1). Patterns, 2025.
