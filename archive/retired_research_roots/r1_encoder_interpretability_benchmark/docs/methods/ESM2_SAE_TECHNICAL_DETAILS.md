# SAEs on ESM-2: Comprehensive Technical Reference for Research Design

## Table of Contents
1. [SAE Training Tools and Frameworks](#1-sae-training-tools-and-frameworks)
2. [Prior Work: InterPLM, InterProt, and Gujral et al.](#2-prior-work-interplm-interprot-and-gujral-et-al)
3. [Protein Annotation Databases](#3-protein-annotation-databases)
4. [ESM-2 Model Architecture Details](#4-esm-2-model-architecture-details)
5. [Evaluation Metrics Used in Bio-SAE Papers](#5-evaluation-metrics-used-in-bio-sae-papers)
6. [Research Gaps and Opportunities](#6-research-gaps-and-opportunities)

---

## 1. SAE Training Tools and Frameworks

### 1.1 SAELens (Bloom, Tigges, Duong, Chanin)

**Repository**: https://github.com/jbloomAus/SAELens (now redirects to decoderesearch/SAELens)
**Docs**: https://decoderesearch.github.io/SAELens/
**Install**: `pip install sae-lens`

**Supported SAE Architectures (v6)**:
| Architecture | Status | Sparsity Mechanism |
|---|---|---|
| BatchTopK | SOTA | Fixes mean L0 across training batch |
| JumpReLU | SOTA | Learnable thresholds with L0-like penalty |
| MatryoshkaBatchTopK | SOTA | Nested reconstruction losses at different widths |
| Standard L1 (ReLU) | Legacy | Classic ReLU + L1 penalty on activations |
| TopK | Legacy | Fixes L0 via TopK activation function |
| Gated | Legacy | Reduces shrinkage vs Standard L1 |
| Matching Pursuit | Research | Iterative latent selection |

**Key Hyperparameters**:
- `d_sae`: SAE hidden dimension (e.g., `d_in * 8` for 8x expansion)
- `l1_coefficient`: Sparsity penalty strength (typical starting point: 8e-5 with lr=4e-4 for GPT2-small)
- `l1_warm_up_steps`: Warmup steps for L1 penalty (helps avoid dead features)
- `k`: Number of active features for TopK/BatchTopK (default: 100)
- `l0_coefficient`: For JumpReLU variant
- `jumprelu_init_threshold`, `jumprelu_bandwidth`: JumpReLU-specific
- `matryoshka_widths`: List of nested widths for Matryoshka variant
- `lr`: Learning rate
- `train_batch_size_tokens`: Batch size in tokens
- `training_tokens`: Total training tokens
- `normalize_activations`: Options include "expected_average_only_in"
- `lr_scheduler_name`: "constant", "cosineannealing", "cosineannealingwarmrestarts"

**Model Support**:
- Native TransformerLens integration (GPT-2, LLaMA, Gemma, Qwen, etc.)
- HuggingFace `AutoModelForCausalLM` via `model_class_name = 'AutoModelForCausalLM'`
- Hook points correspond to named parameters: TransformerLens style (e.g., `blocks.1.hook_resid_post`) or HF style (e.g., `transformer.h.1`)
- **Inference** works with any PyTorch model via `encode()`/`decode()` methods on extracted activations

**Adaptability to ESM-2**:
- SAELens is designed for causal LMs, so direct training integration would require writing a custom activation extraction pipeline
- However, SAE architectures (the `SAE` class) are model-agnostic PyTorch modules
- Best approach: extract ESM-2 activations offline, then use SAELens SAE modules on cached activations
- Or use SAELens as a reference implementation and adapt the training loop

**Pre-cached Activation Training**:
- `use_cached_activations=True` + `cached_activations_path` allows training on pre-extracted activations
- This is the most natural path for non-language models like ESM-2

### 1.2 EleutherAI Sparsify

**Repository**: https://github.com/EleutherAI/sparsify
**Install**: `pip install sparsify`

**Key Features**:
- Lean, simple library with few configuration options
- Does NOT cache activations to disk -- computes on-the-fly
- Scales to very large models with zero storage overhead
- Uses TopK activation function (following Gao et al. 2024)
- Supports both SAEs and transcoders via `--transcode` flag
- Distributed training via `torchrun` with `--distribute_modules` for multi-layer SAEs across GPUs

**Model Support**:
- Works with **any HuggingFace `transformers` model**
- By default trains on residual stream activations
- Pre-trained SAEs available: e.g., `Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")`

**Adaptability to ESM-2**:
- Since ESM-2 is a HuggingFace `transformers` model (`EsmModel`), Sparsify could potentially work with minimal modification
- The main challenge: Sparsify expects causal LM text input pipeline; would need to adapt the data loading for protein sequences
- Most promising framework for direct ESM-2 adaptation due to HF-native design

**Related EleutherAI Tools**:
- **Delphi** (EleutherAI/delphi): Automated interpretability -- generates and scores text explanations of SAE features
- **clt-training** (EleutherAI/clt-training): Cross-layer transcoder training

### 1.3 OpenMOSS Language-Model-SAEs

**Repository**: https://github.com/OpenMOSS/Language-Model-SAEs
**Status**: Active development (Qwen3 support added recently)

**Supported Variants**: Vanilla SAE, Lorsa (Low-rank Sparse Attention), CLT (Cross-layer Transcoder), MoLT, CrossCoder

**Features**:
- Built-in visualization frontend (bun-based, localhost:24576)
- MongoDB integration for configs and analysis storage
- Ascend NPU support
- Forked TransformerLens dependency

**Limitations for ESM-2**: Tightly coupled to TransformerLens, harder to adapt to protein models.

### 1.4 InterPLM (Simon & Zou) -- Purpose-Built for Protein Models

**Repository**: https://github.com/ElanaPearl/InterPLM
**Install**: From source

This is the most directly relevant framework for ESM-2 SAE work.

**Supported SAE Architectures**:
- Standard ReLU SAE
- Top-K SAE
- Jump ReLU SAE
- Batch Top-K SAE

**Code Structure**:
```
interplm/
  sae/          # Model definitions (dictionary.py) & inference
  train/        # Training infrastructure + trainers/
  embedders/    # PLM wrappers (extensible to any PLM)
  analysis/     # Concept association tools
  dashboard/    # Streamlit visualization
```

**Adaptability**: "Primarily set up for ESM-2 embeddings, but can easily be adapted to embeddings from any PLM." The `embedders/` module is designed for extensibility.

### 1.5 Rep_SAEs_PLMs (Gujral & Bafna)

**Repository**: https://github.com/onkarsg10/Rep_SAEs_PLMs

Custom implementation supporting both SAEs and transcoders on ESM-2, with GO enrichment analysis and Claude-based automated interpretation.

### 1.6 Summary: Which Framework to Use

| Framework | ESM-2 Ready? | SAE Variants | Ease of Adaptation | Best For |
|---|---|---|---|---|
| **InterPLM** | Yes (native) | ReLU, TopK, JumpReLU, BatchTopK | Trivial | Direct replication and extension |
| **Sparsify** | Needs modification | TopK, Transcoder | Medium | Large-scale training, transcoders |
| **SAELens** | Needs modification | All 7 variants | Medium-Hard | Using SOTA architectures (Matryoshka, JumpReLU) |
| **Rep_SAEs_PLMs** | Yes (native) | ReLU, Transcoder | Trivial | Protein-level + GO analysis |
| **OpenMOSS** | Hard | Many variants | Hard | Advanced architectures (Lorsa, CrossCoder) |

**Recommendation**: Start with InterPLM for initial experiments. Adapt SAELens architectures (especially Matryoshka, JumpReLU) for novel variants. Use Sparsify for large-scale distributed training.

---

## 2. Prior Work: InterPLM, InterProt, and Gujral et al.

### 2.1 InterPLM (Simon & Zou, Nature Methods 2025)

**Paper**: https://www.nature.com/articles/s41592-025-02836-7
**Preprint**: https://arxiv.org/abs/2412.12101
**Code**: https://github.com/ElanaPearl/InterPLM
**Interactive Explorer**: https://interplm.ai
**Pre-trained SAEs**: HuggingFace `Elana/InterPLM-esm2-8m` and `Elana/InterPLM-esm2-650m`

#### ESM-2 Models Used
- **ESM-2-8M** (`esm2_t6_8M_UR50D`): 6 layers, d_model=320
- **ESM-2-650M** (`esm2_t33_650M_UR50D`): 33 layers, d_model=1280

#### Layers Trained
- ESM-2-8M: All 6 layers (1-6)
- ESM-2-650M: Layers 1, 9, 18, 24, 30, 33 (selected subset)

#### SAE Architecture
- **Type**: Standard ReLU SAE (primary results)
- **Also supports**: TopK, JumpReLU, BatchTopK
- **Dictionary size**: 10,240 for both models
- **Expansion factor**: 32x for ESM-2-8M (320 -> 10,240), 8x for ESM-2-650M (1280 -> 10,240)

#### Training Hyperparameters
- **Training data**: 5 million protein sequences from UniRef50
- **Batch size**: 2,048 tokens
- **Training steps**: 500,000 per SAE
- **Learning rate**: 1e-7 (after sweep from 1e-8 to 1e-4)
- **L1 penalty**: 0.08-0.1 (after sweep from 0.07-0.2)
- **Loss recovery**: 99.6%-100% across layers

#### Normalization
- Feature activations normalized to [0,1] using max activation values from 50,000 Swiss-Prot proteins
- Pre-normalized SAE weights released as `ae_normalized.pt`

#### Key Results
- Up to **2,548 interpretable features** per layer (vs 46 neurons per layer with clear conceptual alignment)
- Features correlate with up to **143 known biological concepts** (vs 15 concepts for neurons)
- Strong evidence for **superposition** in protein language models

#### Novel Feature Discovery Pipeline
- Used **Claude-3.5 Sonnet** to generate automated descriptions of 1,200 features (10% sample)
- Validation: median Pearson r = 0.72 between predicted and actual activation levels
- Performance independent of Swiss-Prot coverage (Pearson r = 0.11 correlation with annotation availability)
- Case studies: identified missing Nudix box motifs, peptidase S1 domains, UDP-GlcNAc binding sites

#### Steering Experiments
- Used NNsight for embedding decomposition
- Successfully steered collagen-like glycine patterns at masked positions
- Effect: "diminishing intensity" propagating across multiple positions
- Limited to "a few, somewhat constrained sequences"

#### Limitations Identified
- Only analyzed ESM-2-8M and ESM-2-650M (not 3B or 15B)
- Excluded masked token and CLS token embeddings
- No scaling to structure-prediction models (ESMFold, AlphaFold)
- No circuit-level analysis combining multiple features
- Steering validation limited to simple patterns
- Current evaluation metrics are approximations of true biological interpretability

#### Biological Annotation Databases Used
- Swiss-Prot (433 concepts evaluated)
- InterPro (for cross-reference validation of missing annotations)

---

### 2.2 InterProt (Adams, Bai, Lee, Yu & AlQuraishi, bioRxiv 2025)

**Paper**: https://www.biorxiv.org/content/10.1101/2025.02.06.636901v2
**Code**: https://github.com/etowahadams/interprot
**Visualizer**: https://interprot.com
**Pre-trained SAEs**: https://huggingface.co/liambai/InterProt-ESM2-SAEs

#### ESM-2 Model Used
- **ESM-2-650M** (`esm2_t33_650M_UR50D`) exclusively

#### Layers Trained
- 8 layers: 4, 8, 12, 16, 20, 24, 28, 32 (evenly spaced across the 33-layer model)

#### SAE Architecture
- **Type**: TopK SAE
- **Encoder**: z = TopK(W_enc(x - b_pre))
- **Decoder**: x_hat = W_dec * z + b_pre
- **Loss**: Reconstruction MSE only (no L1 term -- sparsity enforced structurally by TopK)
- **Hidden dimension**: 4,096 (primary), also tested 8,096
- **Expansion factor**: ~3.2x (1,280 -> 4,096) for primary model
- **k**: 64 (primary), sweeps performed

#### Training Details
- **Training data**: 1 million random sequences under 1,022 residues from UniRef50
- **Embeddings**: Residual stream activations after attention and MLP sublayers
- **Clustering**: Swiss-Prot proteins clustered at 30% sequence identity for evaluation

#### Hyperparameter Sweeps
- **k sweep** (sparsity): Lower k = more family-specific features
- **Hidden size sweep**: 4,096 and 8,096 at layer 24 with k=64
- **Result**: Activation pattern classifications remain largely consistent across expansion factors; family-specific feature count increases

#### Key Results
- ~80% of SAE features received "yes" interpretability rating in human study
- Family-specific features peak in early-to-mid layers, declining in later layers
- Features include: secondary structure, conserved motifs, domains, cysteine bonds, side-chain orientation

#### Activation Pattern Classification
| Category | Description |
|---|---|
| Point | Single residue activations |
| Periodic | Regular intervals, >10 activations per sequence |
| Short Motif | 1-20 residues |
| Medium Motif | 20-50 residues |
| Long Motif | 50-300 residues |
| Domain | >80% sequence coverage |
| Whole | Near-complete sequence activation |

#### Linear Probing Tasks
1. Secondary structure (3-class, residue-level)
2. Subcellular localization (protein-level)
3. Thermostability (Spearman rho)
4. CHO cell expression (binary, novel task)

#### Limitations Identified
- Exclusive focus on 650M variant (no scaling analysis)
- Feature sensitivity to training data distribution
- Mean-pooling limitations for protein-level tasks
- Interpretability constrained by current biological knowledge

---

### 2.3 Gujral et al. (PNAS 2025)

**Paper**: https://www.pnas.org/doi/10.1073/pnas.2506316122
**Code**: https://github.com/onkarsg10/Rep_SAEs_PLMs

#### ESM-2 Model Used
- **ESM-2-35M** (`esm2_t12_35M_UR50D`): 12 layers, d_model=480

#### Representation Levels
- **Protein-level**: Mean-pooled token representations (480-dimensional)
- **Amino acid-level**: Individual token representations (480-dimensional)

#### SAE Architecture
- **Type**: Standard SAE + Transcoder variant
- **Hidden dimension**: 20,000
- **Expansion factor**: ~41.7x (480 -> 20,000)
- **Sparsity (k)**: 64 for protein-level, 16 for amino acid-level

#### Transcoder Details
- Predicts layer-to-layer residual differences (not absolute representations)
- Uses flag "return_difference 1"
- Interpretability on par with standard SAEs

#### Training Details
- **Training data**: UniRef50 (up to 50M samples), excluding Swiss-Prot entries
- **Batch size**: 128
- **Learning rate**: 0.0001
- **Max steps**: 180,000 (protein-level), 70,000 (AA-level)
- **Validation interval**: Every 300 steps
- **Inactivity threshold**: 200 steps (protein), 50 steps (AA)
- **Layer extraction**: Layer 8 (protein-level), Layer 10 (AA-level)
- **Framework**: PyTorch Lightning + W&B

#### Evaluation Methods
- **GO enrichment analysis**: Statistical tests across all GO hierarchy levels
- **Automated interpretation**: Anthropic's Claude for feature narratives
- **Comparison**: Active vs inactive protein sets for each feature
- **Interpretability protocol**: 200 random ESM neurons examined; 8 active + 7 inactive sequences for interpretation, 9 active + 8 inactive for simulation

#### Key Results
- Sparse features tightly associated with GO terms across all hierarchy levels
- Features correspond to specific families: NAD Kinase, IUNH, PTH family
- Sparse features more interpretable than ESM2 neurons across all SAEs and transcoders

#### Biological Annotation Databases Used
- UniRef50 (training data)
- UniProt/Swiss-Prot (evaluation, two different download dates)
- Gene Ontology (GO) terms (all three branches: MF, BP, CC)

#### Limitations
- Only used ESM-2-35M (smallest practical model)
- Only layers 8 and 10 examined
- Limited to single expansion factor

---

### 2.4 Reticular AI (Parsan, Yang & Yang, 2025)

**Paper**: https://arxiv.org/abs/2503.08764
**Visualization**: https://sae.reticular.ai

#### ESM-2 Models Used
- ESM-2-8M and **ESM-2-3B** (first to scale to 3B)

#### SAE Architectures
- TopK SAE
- L1 regularization SAE
- **Matryoshka SAE** (novel for protein models)

#### Layers Trained
- Layers 18 and 36 of ESM-2-3B

#### Key Results
- Scale matters dramatically: 8M -> 3B yields concept coverage jump from ~15-20% to ~49%
- With only 8-32 active latents per token, SAEs reasonably recover structure prediction performance
- Feature steering: shifted myoglobin SASA by +31.5%

#### Limitations
- Only 8M and 3B models analyzed (not 650M for comparison)
- ESMFold no longer SOTA vs diffusion-based methods
- Matryoshka group sizes not systematically explored

---

### 2.5 Comparative Summary Table

| Paper | ESM-2 Size | d_model | SAE Type | Dictionary Size | Expansion | k | Layers | Training Data |
|---|---|---|---|---|---|---|---|---|
| InterPLM | 8M | 320 | ReLU | 10,240 | 32x | N/A (L1) | 1-6 | 5M UniRef50 |
| InterPLM | 650M | 1,280 | ReLU | 10,240 | 8x | N/A (L1) | 1,9,18,24,30,33 | 5M UniRef50 |
| InterProt | 650M | 1,280 | TopK | 4,096 | 3.2x | 64 | 4,8,12,16,20,24,28,32 | 1M UniRef50 |
| Gujral | 35M | 480 | ReLU+TC | 20,000 | 41.7x | 64/16 | 8,10 | 50M UniRef50 |
| Reticular | 3B | 2,560 | TopK/Matr | varies | varies | varies | 18,36 | varies |

---

## 3. Protein Annotation Databases

### 3.1 UniProt / Swiss-Prot

**URL**: https://www.uniprot.org

**Scale (2025)**:
- UniProtKB total: ~246 million sequence records
- Swiss-Prot (manually reviewed): **573,661 entries** (release 2025_04)
- Swiss-Prot covers **<0.25%** of all UniProtKB entries
- TrEMBL (unreviewed): ~245 million entries

**Swiss-Prot Protein Evidence Levels**:
| Evidence Level | Percentage |
|---|---|
| Protein level | 20.7% |
| Transcript level | 9.5% |
| Inferred from homology | 67.3% |
| Predicted | 2.2% |
| Uncertain | 0.3% |

**Annotation Coverage**:
- 207,922,125 amino acids total in Swiss-Prot
- Curated from 306,849 unique references
- ProtNLM has added functional names to 28 million previously "uncharacterized" proteins
- Deep functional annotation remains limited to the Swiss-Prot subset

**Key Annotation Types for SAE Evaluation**:
- Domain annotations (Pfam, InterPro mapped)
- Binding site annotations
- Secondary structure
- Subcellular location
- Post-translational modifications
- Active sites, metal binding, disulfide bonds

### 3.2 Gene Ontology (GO)

**URL**: https://www.geneontology.org

**Scale (April 2024)**:
- 42,255 GO terms
- 7,671,375 annotations
- 1,536,921 gene products
- 5,404 species

**Three Branches**:
1. **Molecular Function (MF)**: Catalytic activity, binding, etc.
2. **Biological Process (BP)**: Metabolic process, signal transduction, etc.
3. **Cellular Component (CC)**: Nucleus, membrane, cytoplasm, etc.

**Annotation Gap**:
- Experimental GO annotations exist for **<0.5% of all known proteins**
- UniRef100 contains ~220 million protein sequences; fewer than 1 million have expert-verified annotations
- Electronic annotations (IEA) cover much more but are lower confidence

**Evidence Codes** (relevant for evaluation quality):
- EXP, IDA, IPI, IMP, IGI, IEP: Experimental evidence (highest quality)
- ISS, ISO, ISA, ISM: Sequence/structural similarity
- IEA: Electronic annotation (lowest confidence but highest coverage)

### 3.3 InterPro

**URL**: https://www.ebi.ac.uk/interpro/

**Integration**: Unifies signatures from 13 member databases:
CATH-Gene3D, CDD, HAMAP, NCBIFAM, PANTHER, PIRSF, **Pfam**, PRINTS, **PROSITE**, SFLD, SMART, SUPERFAMILY

**Coverage**:
- InterPro entries annotated **73.9% of residues** in UniProtKB (as of version 81.0, Aug 2020)
- Additional 9.2% annotated by signatures pending integration
- **Pfam-N** (neural network extension): provides at least one annotation for **85% of sequences** in UniProtKB 2024_04 (8% increase over base Pfam)
- Pfam-N annotates 12,000 Swiss-Prot proteins that Pfam alone does not
- Pfam-N provides annotations for >11 million sequences not annotated by any InterPro member database

### 3.4 Pfam

**URL**: Now hosted by InterPro (http://pfam.xfam.org redirects)

- Large collection of protein families represented by HMMs
- Integrated into InterPro since 2022
- Foundation for many domain annotations used in SAE evaluation

### 3.5 PROSITE

**URL**: https://prosite.expasy.org

- Patterns (regular expressions) and profiles (position-specific scoring matrices)
- Focuses on biologically significant sites, patterns, and domains
- Particularly useful for active sites and post-translational modifications

### 3.6 Annotation Gap Summary

| Database | Scope | Coverage of UniProtKB |
|---|---|---|
| Swiss-Prot (manual) | ~574K proteins | <0.25% |
| GO (experimental) | ~1M gene products | <0.5% |
| InterPro (all) | Residue-level | ~74% of residues |
| Pfam-N (neural) | Sequence-level | ~85% of sequences |
| TrEMBL (automated) | ~245M proteins | ~99.7% (but shallow) |

**Implication for SAE Research**: The vast majority of proteins lack detailed functional annotation. SAE features that do not map to known annotations may represent genuine biological concepts not yet catalogued, making automated interpretation and experimental validation essential.

---

## 4. ESM-2 Model Architecture Details

### 4.1 Full Architecture Table

| Model ID | HuggingFace ID | Params | Layers | d_model | d_ffn | Heads | Head Dim |
|---|---|---|---|---|---|---|---|
| ESM-2-8M | `facebook/esm2_t6_8M_UR50D` | 8M | 6 | 320 | 1,280 | 20 | 16 |
| ESM-2-35M | `facebook/esm2_t12_35M_UR50D` | 35M | 12 | 480 | 1,920 | 20 | 24 |
| ESM-2-150M | `facebook/esm2_t30_150M_UR50D` | 150M | 30 | 640 | 2,560 | 20 | 32 |
| ESM-2-650M | `facebook/esm2_t33_650M_UR50D` | 650M | 33 | 1,280 | 5,120 | 20 | 64 |
| ESM-2-3B | `facebook/esm2_t36_3B_UR50D` | 3B | 36 | 2,560 | 10,240 | 40 | 64 |
| ESM-2-15B | `facebook/esm2_t48_15B_UR50D` | 15B | 48 | 5,120 | 20,480 | 40 | 128 |

**Convention**: d_ffn = 4 * d_model for all sizes. Heads increase from 20 to 40 at 3B+.

### 4.2 Where to Extract Activations

ESM-2 uses a standard encoder-only transformer architecture. Activation extraction points:

| Hook Point | Description | Dimension | Used By |
|---|---|---|---|
| **Residual stream (post-layer)** | Output after full transformer block | d_model | InterProt (primary) |
| **Residual stream (mid-layer)** | After attention, before MLP | d_model | Possible but not yet explored |
| **MLP output** | Feed-forward network output | d_model | Possible (standard for NLP SAEs) |
| **Attention output** | Multi-head attention output | d_model | Not yet explored for proteins |
| **MLP hidden** | FFN intermediate activations | d_ffn (4x d_model) | Transcoders (Gujral) |
| **Mean-pooled representation** | Average across sequence positions | d_model | Gujral (protein-level) |
| **CLS token** | Special classification token | d_model | Not used in prior SAE work |

**Notes**:
- InterPLM and InterProt both train on residual stream activations
- Gujral et al. extract at specific layers (8 or 10) and mean-pool for protein-level
- No prior work has systematically compared extraction points (residual vs MLP vs attention)
- Max sequence length: 1,022 amino acids (longer sequences are truncated)

### 4.3 Computational Requirements

**Inference** (approximate, single A100 80GB):

| Model | GPU Memory | Throughput (seq/sec, len=512) |
|---|---|---|
| ESM-2-8M | ~1 GB | ~1,000+ |
| ESM-2-35M | ~1 GB | ~500+ |
| ESM-2-150M | ~2 GB | ~200+ |
| ESM-2-650M | ~3 GB | ~50-100 |
| ESM-2-3B | ~12 GB | ~10-20 |
| ESM-2-15B | ~60 GB (or multi-GPU) | ~1-5 |

**SAE Training** (estimated based on prior work):
- InterPLM (8M, 32x): Hours on single GPU
- InterPLM (650M, 8x): Days on single GPU
- InterProt (650M, 3.2x): Days on single GPU
- Full pipeline (extract embeddings + train SAE + evaluate): 1-3 days per layer on A100

### 4.4 HuggingFace Loading

```python
from transformers import AutoModel, AutoTokenizer

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Extract residual stream activations at all layers
inputs = tokenizer("MKTLLILAVL", return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states  # Tuple of (n_layers+1,) tensors of shape (batch, seq_len, d_model)
# hidden_states[0] = embedding layer output
# hidden_states[i] = output of transformer layer i (1-indexed)
```

---

## 5. Evaluation Metrics Used in Bio-SAE Papers

### 5.1 InterPLM Evaluation Framework

#### Automated Concept Alignment (Primary Metric)
- **Database**: Swiss-Prot annotations (433 concepts evaluated)
- **Method**: Domain-level F1 score
  - Precision: calculated per amino acid (what fraction of activated residues fall within annotated domains)
  - Recall: calculated per domain (what fraction of annotated domains are detected)
  - This handles features that activate on subsets of larger domains without penalizing them unfairly
- **Result**: Features with F1 > threshold are considered "aligned" with a concept
- **Example**: Feature f/1503 has F1=0.998 as a TBDR detector

#### Randomized Baselines
- SAE with expansion factor 1x and identity encoder/decoder (neuron baseline)
- Randomly initialized SAE (shuffled baseline)
- 6 hyperparameter choices per layer for baseline comparisons

#### LLM-Based Automated Interpretation
- Model: Claude-3.5 Sonnet (new)
- Sample: 1,200 features (10% of total)
- Validation: Pearson r = 0.72 median between predicted and actual activation levels
- Key finding: LLM interpretation quality is independent of existing annotation coverage (r = 0.11)

#### Missing Annotation Detection
- Pipeline: Identify proteins with high SAE feature activation but missing Swiss-Prot annotation
- Validation: Cross-reference with InterPro database
- Case studies: Q7JIG6 (peptidase S1 domain), Nudix box motifs, UDP-GlcNAc binding sites

#### Reconstruction Metrics
- Loss recovery: 99.6-100% across layers
- Measured per-layer for each SAE

### 5.2 InterProt Evaluation Framework

#### Human Interpretability Study
- **Participants**: 7 (graduate students + undergraduates with protein biology familiarity)
- **Protocol**: 100 randomly selected features per participant, blinded study
- **Rating**: 3-point scale (yes / no / maybe)
- **Result**: ~80% of SAE features received "yes" rating
- **Comparison**: SAE latents consistently rated as interpretable; ESM baseline latents were not

#### Family Specificity
- **Method**: Binary classification on Swiss-Prot proteins (clustered at 30% sequence identity)
- **Threshold**: F1 > 0.7 for "family-specific" designation
- **Ground truth**: InterPro family annotations
- **Finding**: Lower k (more sparsity) increases family-specific features; middle layers contain most family-specific features

#### Activation Pattern Classification (Rule-Based)
| Pattern | Criteria |
|---|---|
| Point | Single residue activations |
| Periodic | Regular intervals, >10 activations/sequence |
| Short Motif | 1-20 consecutive residues |
| Medium Motif | 20-50 consecutive residues |
| Long Motif | 50-300 consecutive residues |
| Domain | >80% sequence coverage |
| Whole | Near-complete sequence activation |

#### Linear Probing
- **Tasks**: Secondary structure (3-class), subcellular localization, thermostability (Spearman rho), CHO cell expression (binary)
- **Method**: Grid search over regularization strengths; mean-pooling for protein-level tasks
- **Purpose**: Demonstrates that SAE features retain and organize biologically relevant information

### 5.3 Gujral et al. Evaluation Framework

#### GO Enrichment Analysis
- Statistical tests for association between each feature and all GO terms
- Evaluated across all three GO branches (MF, BP, CC)
- Tests applied to protein-level and AA-level features independently

#### Automated Interpretation with Claude
- **Protocol**: Select active and inactive sequences per feature
- **Sequences**: 8 active + 7 inactive for interpretation; 9 active + 8 inactive for simulation
- **Sample**: 200 random ESM neurons for comparison
- **Finding**: SAE features more interpretable than ESM2 neurons

#### Transcoder Comparison
- Transcoders achieve interpretability on par with SAEs
- Provides alternative view: understanding layer-to-layer information transformation

### 5.4 Reticular AI Evaluation

#### Concept Coverage
- Measure: fraction of known biological concepts recovered by SAE features
- Key finding: ESM-2-3B recovers ~49% of concepts vs ~15-20% for ESM-2-8M

#### Structure Prediction Recovery
- Metric: How well does SAE-reconstructed representation recover ESMFold contact maps?
- Finding: 8-32 active latents per token are sufficient for reasonable structure prediction

#### Feature Steering
- Metric: Change in predicted structural properties (e.g., SASA change of +31.5% for myoglobin)

### 5.5 Summary of Evaluation Approaches

| Metric | InterPLM | InterProt | Gujral | Reticular |
|---|---|---|---|---|
| Domain-level F1 | Yes (primary) | No | No | No |
| Human evaluation | No | Yes (7 raters) | No | No |
| Family specificity F1 | No | Yes (primary) | No | No |
| GO enrichment | No | No | Yes (primary) | No |
| LLM auto-interp | Yes (Claude) | No | Yes (Claude) | No |
| Linear probing | No | Yes (4 tasks) | No | No |
| Activation patterns | No | Yes (7 categories) | No | No |
| Reconstruction loss | Yes | Yes | Yes | Yes |
| Concept coverage % | Yes | No | No | Yes |
| Structure prediction | No | No | No | Yes |
| Feature steering | Yes (limited) | No | No | Yes |

---

## 6. Research Gaps and Opportunities

### 6.1 Identified Gaps from Prior Work

1. **Model scale**: No systematic comparison across all ESM-2 sizes (only 8M, 35M, 650M, and 3B individually studied)
2. **Layer coverage**: Most work focuses on specific layers; no comprehensive all-layer analysis for 650M or 3B
3. **SAE architecture comparison**: No head-to-head comparison of ReLU vs TopK vs JumpReLU vs BatchTopK vs Matryoshka on the same model/layer
4. **Activation extraction point**: All prior work uses residual stream; MLP output and attention output are unexplored
5. **Circuit analysis**: No feature-to-feature interaction analysis (analogous to Anthropic's attribution graphs)
6. **Cross-layer feature tracking**: No systematic study of how features evolve across layers (CrossCoder-style)
7. **Transcoder scaling**: Only applied to ESM-2-35M; not tested on larger models
8. **Evaluation standardization**: Each paper uses different metrics; no unified benchmark
9. **Novel biology validation**: Most "novel" features are validated by cross-referencing databases, not by experimental biology
10. **Structure prediction integration**: Only Reticular AI connected SAE features to ESMFold; no work on AlphaFold or ESM3
11. **ESM-2-150M**: Completely unexplored by any SAE paper (sweet spot for cost-efficiency?)
12. **Multi-modal features**: No integration with structural features from PDB or ligand binding data
13. **Evolutionary analysis**: No systematic study of how features relate to evolutionary conservation (e.g., conservation scores, phylogenetic patterns)
14. **Disease-relevant features**: No connection to disease variants (ClinVar, HGMD) or drug targets

### 6.2 Opportunities for Novel Contributions

1. **Scaling laws for protein SAEs**: How does interpretability scale with model size x expansion factor x sparsity?
2. **Cross-layer transcoders on ESM-2-650M**: Bring CLT (from Anthropic's circuit tracing) to protein models
3. **Automated biological discovery pipeline**: SAE features -> LLM interpretation -> experimental hypothesis -> validation
4. **Feature-based protein function prediction**: Use SAE features as a new embedding for downstream tasks
5. **Dark proteome illumination**: Systematically analyze features activating on proteins with no known function
6. **Evolutionary feature tracking**: How do SAE features correlate with evolutionary rates, conservation, and selection pressure?
7. **Disease variant sensitivity**: Do specific SAE features respond to pathogenic vs benign mutations?

---

## References

### Primary Papers
1. Simon E, Zou J. "InterPLM: discovering interpretable features in protein language models via sparse autoencoders." Nature Methods (2025). https://www.nature.com/articles/s41592-025-02836-7
2. Adams E, Bai L, Lee M, Yu Y, AlQuraishi M. "From Mechanistic Interpretability to Mechanistic Biology." bioRxiv (2025). https://www.biorxiv.org/content/10.1101/2025.02.06.636901v2
3. Gujral O, Bafna M, Alm E, Berger B. "Sparse autoencoders uncover biologically interpretable features in protein language model representations." PNAS (2025). https://www.pnas.org/doi/10.1073/pnas.2506316122
4. Parsan P, Yang K, Yang KK. "Towards Interpretable Protein Structure Prediction with Sparse Autoencoders." arXiv (2025). https://arxiv.org/abs/2503.08764

### SAE Tools
5. Bloom J et al. "SAELens." (2024). https://github.com/jbloomAus/SAELens
6. EleutherAI. "Sparsify." https://github.com/EleutherAI/sparsify
7. OpenMOSS. "Language-Model-SAEs." https://github.com/OpenMOSS/Language-Model-SAEs

### Protein Models
8. Lin Z et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science (2023). https://github.com/facebookresearch/esm

### Databases
9. UniProt Consortium. "UniProt: the Universal Protein Knowledgebase in 2025." NAR (2025). https://academic.oup.com/nar/article/53/D1/D609/7902999
10. Paysan-Lafosse T et al. "InterPro: the protein sequence classification resource in 2025." NAR (2025). https://academic.oup.com/nar/article/53/D1/D444/7905301
11. Gene Ontology Consortium. https://www.geneontology.org/
