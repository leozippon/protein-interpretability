# Sparse Autoencoders as Protein Microscopes: Interpretable Features for Functional Annotation Beyond Sequence Homology

## Final Research Plan (v2.1)

**Project Codename**: BioInterpretability
**Date**: 2026-03-24 (revised from v2)
**Target venue**: Nature Methods

---

## I. Research Overview

### 1.1 Core Thesis

Protein language models (PLMs) trained on evolutionary data encode biological knowledge far beyond current human annotation. We use Sparse Autoencoders (SAEs) to decompose ESM-2-3B's internal representations into interpretable features, then demonstrate that these features constitute a **new functional annotation system** — one that operates through mechanistic decomposition rather than sequence homology, and can annotate the "dark proteome" where homology-based methods fail.

We further characterize **how biological information emerges across model depth** — showing when different types of knowledge (local motifs → secondary structure → domains → function) become decodable, and how SAE features at different layers capture a biological information hierarchy.

### 1.2 One-Sentence Summary

> "We decompose a 3-billion-parameter protein language model into interpretable features that function as a new annotation system for the dark proteome, and reveal how biological knowledge is organized across the model's depth."

### 1.3 Why This Work Is Different

| Existing work | What they did | Gap we fill |
|---------|------------|-------------|
| **InterPLM** (Nature Methods 2025) | Trained ReLU SAEs on ESM-2-8M/650M; found features matching 143 known concepts | No benchmark proving features are *useful* as an annotation tool; no cross-layer analysis; no dark proteome application |
| **InterProt** (ICML 2025 Spotlight) | Trained TopK SAEs on ESM-2-650M; built visualization tool | Descriptive analysis only; no prediction benchmark; no layer-depth analysis |
| **Gujral et al.** (PNAS 2025) | SAE + Transcoder on ESM-2-35M; GO enrichment | Small model; no systematic annotation benchmark; no depth analysis |
| **Reticular AI** (arXiv 2025) | First SAEs on ESM-2-3B; feature steering | Only 2 layers; no annotation benchmark; no biological validation |

### 1.4 Core Contributions (Ordered by Importance)

1. **SAE features as a functional annotation system**: Rigorous benchmark demonstrating that SAE features can recover held-out protein annotations, competing with and complementing InterProScan/BLAST/HMMs — especially on proteins where homology-based methods fail
2. **Cross-layer biological information hierarchy**: First systematic characterization of when and where different types of biological knowledge emerge across an encoder PLM's depth, using SAE features + linear probes at 9 layers to reveal the information hierarchy (local chemistry → secondary structure → domains → function)
3. **Dark proteome annotation transfer**: Apply SAE-based annotation to ~200M uncharacterized TrEMBL proteins and ~4,000 DUF (Domain of Unknown Function) families, generating testable functional hypotheses
4. **Predictive validation**: SAE features predict variant pathogenicity, binding sites, and PTM locations — with interpretable explanations that black-box methods cannot provide
5. **Multi-level biological validation of novel features**: Systematic pipeline (conservation + 3D structure + disease variants + experimental cross-check) for features that match no known annotation

---

## II. Model and SAE Architecture

### 2.1 Compute Resources

| Spec | Value |
|------|-------|
| **GPU** | 8 × NVIDIA L20 (48 GB VRAM each) |
| **Total VRAM** | 384 GB |
| **FP16 throughput** | ~119 TFLOPS per card |

### 2.2 Primary Model: ESM-2-3B

`facebook/esm2_t36_3B_UR50D` — 36 layers, d=2560, 40 heads.

**Why 3B**: Larger models encode more biological knowledge (Reticular AI showed ~49% concept coverage at 3B vs ~15-20% at 8M). The additional representational capacity means more features that go *beyond* current annotation — the core substrate for our annotation system.

### 2.3 Comparison Model: ESM-2-650M

`facebook/esm2_t33_650M_UR50D` — reuse InterProt pre-trained SAE as baseline for cross-scale comparison.

### 2.4 SAE Architecture: Per-Example TopK

| Hyperparameter | Value | Rationale |
|-------|-----|---------|
| **Architecture** | Per-example TopK | Validated by InterProt; BatchTopK causes excessive dead features on protein SAEs |
| **d_sae** | 16,384 | 6.4× expansion for layers 3-19; consider 32,768 for deeper layers |
| **k** | 256 | k/d_sae=1.56%; doubled from initial 128 for better reconstruction |
| **Normalization** | Anthropic sqrt(d) | Preserves direction while equalizing magnitude across layers; critical for deeper ESM-2 layers where activation norms grow ~100× |
| **Dead feature mitigation** | Auxk loss (α=0.25) + resampling every 25K steps | Addresses gradient dead zones; ReLU applied to auxk values |
| **Training data** | 5M sequences from UniRef50 | Matches InterPLM |
| **Training steps** | 500,000 per layer | Matches InterPLM |
| **Layers** | [3, 7, 11, 15, 19, 23, 27, 31, 35] | 9 layers spanning full depth |

### 2.5 Technical Innovations in SAE Training

Three improvements over prior protein SAE work, discovered during EXP-001–EXP-005:

1. **Sqrt(d) normalization** (from InterPLM/Anthropic): Replaces LayerNorm (InterProt) or no normalization. Normalizes each token to unit L2 norm then scales by √d_in, preserving activation direction while equalizing magnitude. Critical result: without normalization, deeper layer loss explodes ~4000× due to activation magnitude scaling.

2. **b_dec initialization in normalized space**: When using sqrt(d) normalization, the decoder bias must be initialized from the mean of *normalized* activations, not raw activations. Failure to do this causes ~3986× loss inflation and >98% dead features (discovered and fixed in this project).

3. **Auxk ReLU fix**: The auxiliary dead-feature loss must apply ReLU to selected pre-activations before decoding, matching the main path behavior. Without this, negative activations produce garbage reconstructions that corrupt the auxk gradient signal.

---

## III. Experimental Design

### Phase 1: SAE Training and Quality Assurance (Weeks 1–5) [IN PROGRESS]

#### 1.1 Training Pipeline
```
UniRef50 batch → ESM-2-3B forward (fp16) → extract layer activations
    → sqrt(d) normalize → SAE forward/backward → update SAE weights
```

Online training (no activation caching) avoids ~80TB disk requirement.

#### 1.2 Quality Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| FVU (Fraction of Variance Unexplained) | < 0.05 | Better than InterPLM's reported values |
| Dead features | < 5% | Matches InterProt |
| L0 (active features per token) | 256 (= k) | By construction |

#### 1.3 Current Status

| Layer | Status | FVU | Dead % | Notes |
|-------|--------|-----|--------|-------|
| 3 | Complete (500K steps) | 0.087 | 4.8% | Trained without normalization |
| 7 | Complete (500K steps) | 0.167 | 6.2% | Without normalization |
| 11 | Complete (500K steps) | 0.203 | 7.1% | Without normalization |
| 15 | Complete (500K steps) | 0.266 | 8.3% | Without normalization |
| 19 | **Training** (~step 7K) | ~0.45 | 0% | WITH sqrt(d) norm + k=256 |
| 23–35 | Queued | — | — | Will use sqrt(d) norm + k=256 |

**Note**: Layers 3–15 should be retrained with sqrt(d) normalization for consistency before the annotation benchmark. The normalization fix is expected to significantly improve FVU.

---

### Phase 2: Annotation Benchmark — SAE Features as a Functional Annotation System (Weeks 5–9)

**This is the paper's central contribution.** We rigorously benchmark whether SAE features can function as a protein annotation tool.

#### 2.1 Benchmark Design

**Held-out evaluation protocol:**
```
1. Full Swiss-Prot dataset (~574K proteins with rich annotations)
2. Split annotations (NOT proteins) into:
   - Training set (80%): used to learn feature→annotation mappings
   - Test set (20%): held out for blind evaluation
3. For each SAE feature, compute activation patterns on Swiss-Prot proteins
4. Learn lightweight classifiers: SAE features → annotation labels
5. Evaluate on held-out annotations
```

**Why split annotations, not proteins**: ESM-2 was trained on all of UniRef (which includes Swiss-Prot sequences), so there is no clean protein-level split. Instead, we hold out specific annotation *types* or *instances* — e.g., train on domain annotations, test on binding site recovery; or hold out 20% of each annotation type's instances.

#### 2.2 Annotation Types to Benchmark

| Annotation type | Source | Scale | Why it matters |
|----------------|--------|-------|----------------|
| Protein domains | Pfam/InterPro | ~19,000 families | Core structural/functional units |
| Binding sites | Swiss-Prot + BioLiP | ~500K sites | Drug target relevance |
| Active sites | Swiss-Prot | ~200K sites | Enzyme function |
| PTM sites | PhosphoSitePlus + Swiss-Prot | ~450K sites | Signaling regulation |
| Secondary structure | DSSP (from PDB) | Millions of residues | Fundamental structure |
| Transmembrane regions | Swiss-Prot + TMHMM | ~100K proteins | Membrane biology |
| Signal peptides | Swiss-Prot + SignalP | ~60K proteins | Protein targeting |
| Disorder regions | DisProt + MobiDB | ~30K proteins | Intrinsically disordered |
| GO terms (MF/BP/CC) | UniProt-GOA | ~7.7M annotations | Broad function |

#### 2.3 Baselines for Comparison

| Method | Type | What it tests |
|--------|------|---------------|
| **InterProScan** | Homology-based (HMMs) | Gold standard for domain annotation |
| **BLAST / PSI-BLAST** | Sequence similarity | Simplest transfer method |
| **Linear probes on ESM-2** | Representation learning | Tests whether SAE decomposition adds value over raw embeddings |
| **k-NN in ESM-2 embedding space** | Non-parametric | Another embedding baseline |
| **Random SAE directions** | Negative control | Tests whether learned features beat random projections |
| **Untrained SAE** | Critical control | Tests whether signal comes from SAE training vs. geometric artifacts |
| **InterProt SAE (650M)** | Cross-scale comparison | Direct comparison with published method |

#### 2.4 Metrics

Following CAFA (Critical Assessment of Functional Annotation) standards:
- **Residue-level**: Precision, Recall, F1, AUPRC for each annotation type
- **Protein-level**: F-max, S-min (semantic similarity) for GO terms
- **Coverage**: Fraction of proteins receiving at least one annotation above threshold

#### 2.5 Key Hypothesis

**SAE features will outperform homology-based methods on "orphan" proteins** — proteins with no close homologs in Swiss-Prot (sequence identity <30%). This is because SAEs capture abstract patterns learned from evolutionary data, not direct sequence similarity.

To test: stratify benchmark results by sequence identity to nearest Swiss-Prot hit. Show that SAE annotation quality degrades more gracefully than BLAST/InterProScan as homology decreases.

#### 2.6 Annotation Transfer Protocol

```
For each SAE feature f_i:
  1. Compute activation threshold (95th percentile of nonzero activations)
  2. Binarize: feature "active" when activation > threshold
  3. For each annotation type a_j:
     - Compute precision/recall/F1 between f_i activation and a_j labels
     - If F1 > 0.3: associate f_i with a_j
  4. For unannotated residues where f_i activates:
     - Transfer annotation a_j with confidence = activation strength
```

---

### Phase 3: Cross-Layer Biological Information Hierarchy (Weeks 7–10)

**First systematic characterization of how biological knowledge is organized across depth in a protein language model.**

#### 3.1 Motivation

All prior protein SAE work analyzes features at a single layer in isolation. But proteins are hierarchical: local amino acid chemistry → secondary structure → domains → tertiary fold → function. Does the model's representational hierarchy mirror this biological hierarchy? Understanding *when* different types of biological information emerge across depth is both scientifically interesting and practically important (it tells users which layer's SAE features to use for which annotation task).

#### 3.2 Method A: Linear Probing Across Layers (Well-Established)

For each layer L ∈ [3, 7, 11, 15, 19, 23, 27, 31, 35]:

```
Train lightweight linear probes to predict:
  1. Amino acid identity (sanity check — should be decodable from layer 0)
  2. Secondary structure (3-class: helix/sheet/coil — from DSSP)
  3. Solvent accessibility (buried vs exposed — from DSSP)
  4. Domain membership (Pfam domain labels)
  5. Binding site residues (from BioLiP)
  6. Conservation score (ConSurf)
  7. Contact number (from AlphaFold structures)
  8. GO term (protein-level molecular function)
```

Plot accuracy vs layer depth → reveals the "information creation curve" for each biological property. Expected: secondary structure peaks early-mid, domains mid-late, function late.

#### 3.3 Method B: SAE Feature Annotation Across Layers (Novel)

At each layer, use the Phase 2 annotation benchmark to characterize which annotation types are best captured:

```
For each layer L:
  1. Run Swiss-Prot through ESM-2 → SAE at layer L
  2. Compute feature↔annotation F1 for all annotation types
  3. Count features classified as KNOWN for each annotation type
  4. Record the best-F1 feature for each annotation type

Result: a "layer × annotation type" heatmap showing where each
type of biological knowledge is most strongly represented as
distinct SAE features.
```

This goes beyond linear probes: probes show what's *decodable* (could be distributed across many neurons), while SAE analysis shows what's *represented as discrete features* (localized, interpretable units).

#### 3.4 Method C: Feature Specialization Analysis

Track how features change qualitatively across depth:

```
Layer 3-7 features: expected to capture local sequence patterns
  → Amino acid motifs, charge patterns, hydrophobic clusters
  → Metric: average activation span (positions activated per protein)
     should be SHORT (3-10 residues)

Layer 11-19 features: expected to capture secondary structure + local structure
  → Helices, sheets, loops, turns
  → Metric: activation span should be MEDIUM (10-50 residues)

Layer 23-31 features: expected to capture domains + global structure
  → Domain boundaries, structural families
  → Metric: activation span should be LONG (50-200 residues)

Layer 35 features: expected to capture function + whole-protein properties
  → Enzyme families, subcellular localization
  → Metric: protein-level features (>50% of residues activated)
```

#### 3.5 Deliverables

1. **Information emergence curve** (Fig 3a): Probe accuracy vs layer depth for 8 biological properties
2. **Layer × annotation heatmap** (Fig 3b): Which SAE features capture which annotations at which depth
3. **Feature granularity progression** (Fig 3c): Average activation span from local (early) to global (late)
4. **Practical guide**: "Use layer X's SAE for binding site annotation, layer Y's for domain annotation" — directly actionable for downstream users

---

### Phase 4: Dark Proteome Annotation (Weeks 9–13)

#### 4.1 The Opportunity

| Category | Scale | Current state |
|----------|-------|---------------|
| **TrEMBL uncharacterized proteins** | ~200M proteins | No functional annotation beyond automated predictions |
| **DUF families** (Domains of Unknown Function) | ~4,000 Pfam entries | Named families with no known function |
| **Dark kinases** | ~160 human proteins | Kinases with no known substrates (IDG consortium priority) |
| **Orphan GPCRs** | ~120 human proteins | Receptors with no known ligands |

#### 4.2 Annotation Transfer Pipeline

```
1. Run ESM-2 → SAE inference on target proteins
2. For each protein:
   a. Identify which SAE features activate at each residue
   b. Look up each feature's annotation associations (from Phase 2 benchmark)
   c. Transfer annotations with confidence scores
   d. Flag novel feature activations (no known annotation match)
3. Aggregate: protein-level functional predictions
4. Validate predictions against:
   - Recent Swiss-Prot updates (annotations added AFTER our training cutoff)
   - Literature mining (PubMed abstracts mentioning the protein)
   - AlphaFold predicted structures (functional site plausibility)
```

#### 4.3 Focused Case Studies

**Dark kinases** (highest impact):
- ~160 human kinases with no known substrates or structures
- IDG (Illuminating the Druggable Genome) consortium has prioritized these
- SAE features may reveal: substrate binding sites, allosteric sites, activation loop features
- Validate against the few that have been characterized since IDG started

**DUF families** (broadest impact):
- Select 20 DUF families with the most consistent SAE feature patterns
- Cluster DUF family members by SAE feature activation similarity
- Generate functional hypotheses: "DUF1234 members share SAE features associated with metal binding + membrane association → hypothesis: metal-dependent membrane transporter"
- Cross-validate with AlphaFold structure predictions

#### 4.4 Temporal Validation

**Critical control**: Use Swiss-Prot version history to test whether SAE-based annotations predict annotations that were *added* in later database releases.
```
1. Train annotation mappings on Swiss-Prot release 2024_01
2. Generate predictions for proteins with missing annotations
3. Check which predictions are confirmed by Swiss-Prot release 2025_01
4. This demonstrates genuine predictive power, not post-hoc fitting
```

---

### Phase 5: Predictive Tasks — Proving Utility (Weeks 10–14)

SAE features must demonstrate utility on concrete prediction tasks that biologists care about.

#### 5.1 Variant Effect Prediction

```
Task: Predict whether a missense variant is pathogenic
Data: ClinVar pathogenic + gnomAD benign variants (human proteome)
Method:
  1. For each variant position, extract SAE feature activations
  2. Train lightweight classifier: SAE features → pathogenic/benign
  3. Compare against:
     - AlphaMissense (SOTA, Google DeepMind)
     - EVE (evolutionary model)
     - ESM-1v (PLM-based)
     - CADD, REVEL (ensemble methods)
Unique advantage: SAE-based predictions are INTERPRETABLE
  - "This variant is predicted pathogenic because it disrupts Feature #7234
    (associated with catalytic triad positioning in serine proteases)"
  - No other method provides this level of mechanistic explanation
```

#### 5.2 Binding Site Prediction

```
Task: Predict ligand-binding residues
Data: BioLiP database (experimentally determined ligand contacts)
Method: SAE features → binding site classifier
Compare against: P2Rank, FPocket, ScanNet
```

#### 5.3 Protein-Protein Interaction Interface Prediction

```
Task: Predict PPI interface residues
Data: PDB biological assemblies (interface residues within 6Å)
Method: SAE features → interface classifier
Compare against: PIONEER, ECLAIR, AlphaFold-Multimer
```

#### 5.4 Key Metric: Performance on Orphan Proteins

For all three tasks, stratify results by homology level:
- **>50% identity** to training proteins (easy)
- **30-50% identity** (moderate)
- **<30% identity** (hard — "twilight zone")
- **<20% identity** (very hard — "midnight zone")

**Hypothesis**: SAE features maintain predictive accuracy in the twilight/midnight zone where homology-based methods fail, because SAEs capture abstract biological patterns rather than sequence similarity.

---

### Phase 6: Novel Feature Validation (Weeks 11–15)

For features that match no known annotation (F1 < 0.2 with all Swiss-Prot types), validate biological reality through multiple orthogonal lines of evidence.

#### 6.1 Critical Controls (MUST DO)

| Control | Purpose | Expected result |
|---------|---------|-----------------|
| **Untrained SAE** | Test if enrichment comes from SAE training vs geometric artifacts | No significant enrichment |
| **Random directions in activation space** | Test if signal is specific to learned features | No significant enrichment |
| **Activation shuffling** (shuffle across residues) | Test if signal comes from positions, not feature identity | Enrichment disappears |
| **Dead features** | Verify inactive features carry no signal | No enrichment |
| **Known-feature calibration** | Establish effect-size baseline for real biology | Provides reference values |

#### 6.2 Validation Layers

1. **Evolutionary conservation** (ConSurf): activated residues should be more conserved than random
2. **3D spatial clustering** (AlphaFold): activated residues should cluster in 3D space
3. **Disease variant enrichment** (ClinVar/gnomAD): activated residues enriched for pathogenic variants
4. **Experimental cross-check** (BioLiP, PhosphoSitePlus, PPI databases): overlap with functional sites
5. **Cross-family consistency**: feature activates across unrelated protein families (rules out homology artifact)

#### 6.3 Deep Case Studies

Select **3–5 most compelling novel features** for deep analysis:
- Coevolution analysis (EVcouplings)
- Structural environment (pocket depth, solvent accessibility)
- Cross-species conservation in orthologs
- Literature mining for unpublished experimental evidence
- Generate explicit, experimentally testable hypotheses

**Priority targets**: Novel features that activate on dark kinases or DUF family proteins — these have the highest biological impact and community interest.

---

### Phase 7: Ablation Studies (Weeks 13–16)

| Experiment | Purpose | Settings |
|------------|---------|----------|
| **Expansion factor** | How dictionary size affects annotation quality | d_sae = 8192, 16384, 32768 on layer 23 |
| **Sparsity (k)** | How sparsity affects feature granularity | k = 64, 128, 256, 512 on layer 23 |
| **Layer depth** | Which layers are most useful for annotation | Compare annotation benchmark across all 9 layers |
| **Model scale** | Does 3B annotate better than 650M? | ESM-2-650M vs 3B on same layers/tasks |
| **Normalization** | Impact of sqrt(d) vs LayerNorm vs none | Three variants on layer 23 |
| **SAE architecture** | TopK vs ReLU+L1 | On layer 23 (BatchTopK excluded due to dead features) |

---

## IV. Datasets

### 4.1 Training Data

| Dataset | Purpose | Scale |
|---------|--------|-------|
| **UniRef50** | SAE training | 5M sequences (sampled) |

### 4.2 Evaluation Data

| Dataset | Purpose | Scale |
|---------|--------|-------|
| **Swiss-Prot** | Annotation benchmark (80/20 split) | ~574K proteins |
| **InterPro** | Domain/motif cross-validation | Covers 74% of residues |
| **Gene Ontology (GOA)** | GO term prediction | ~7.7M annotations |
| **ClinVar** | Variant pathogenicity | ~2.5M variants |
| **gnomAD v4.1** | Benign variant control | 730K exomes |
| **BioLiP** | Ligand binding sites | ~200K structures |
| **PhosphoSitePlus** | PTM sites | >450K sites |
| **PDB (DSSP)** | Secondary structure ground truth | ~200K structures |
| **AlphaFold DB** | 3D structure mapping | 200M+ structures |
| **ConSurf-DB** | Conservation scores | All PDB structures |
| **DisProt / MobiDB** | Disorder regions | ~30K proteins |

### 4.3 Target Protein Sets

| Set | Purpose | Scale |
|-----|---------|-------|
| **Human proteome** | Best ClinVar/gnomAD coverage; variant prediction | ~20K proteins |
| **Dark kinases** | Highest-impact annotation transfer target | ~160 proteins |
| **DUF families** | Broadest annotation transfer | ~4,000 families |
| **Orphan GPCRs** | Drug target annotation | ~120 proteins |
| **TrEMBL uncharacterized** | Full dark proteome application | ~200M proteins (sample) |

---

## V. Paper Structure (Draft Outline)

### Title Options
1. "Mechanistic Decomposition of Protein Language Models Enables Functional Annotation of the Dark Proteome"
2. "Sparse Autoencoders as Protein Microscopes: Interpretable Features for Functional Annotation Beyond Sequence Homology"

### Figures Plan

**Figure 1**: Overview + SAE method
- (a) Schematic: ESM-2 → SAE decomposition → interpretable features → annotation
- (b) SAE training quality across layers (FVU, dead %, reconstruction examples)
- (c) Feature classification pie chart: KNOWN / PARTIAL / NOVEL across layers

**Figure 2**: Annotation benchmark (THE KEY FIGURE)
- (a) Head-to-head: SAE annotation vs InterProScan vs BLAST vs linear probes, stratified by homology level
- (b) Performance degradation curve: accuracy vs sequence identity to nearest Swiss-Prot hit — SAE degrades slower
- (c) Annotation types where SAE outperforms (binding sites, PTMs) vs where homology wins (domains)
- (d) Temporal validation: SAE predictions confirmed by later Swiss-Prot releases

**Figure 3**: Cross-layer biological information hierarchy
- (a) Information emergence curves: probe accuracy vs layer depth for 8 biological properties
- (b) Layer × annotation type heatmap: which SAE features capture which biology at which depth
- (c) Feature granularity progression: activation span from local motifs (early layers) to global function (late layers)
- (d) Practical layer selection guide for downstream annotation tasks

**Figure 4**: Dark proteome annotation
- (a) Annotation transfer coverage: how many previously unannotated proteins/residues gain annotations
- (b) Dark kinase case study: SAE features reveal binding site on an uncharacterized kinase
- (c) DUF family case study: SAE features suggest function for a domain of unknown function
- (d) Confidence calibration: are high-confidence SAE annotations more likely to be correct?

**Figure 5**: Novel feature validation (THE "KILLER FIGURE")
- One compelling novel feature, validated from 5 orthogonal angles:
  - (a) Activation pattern on 3D structures of representative proteins
  - (b) Conservation enrichment (ConSurf)
  - (c) Disease variant enrichment (ClinVar)
  - (d) Spatial clustering in AlphaFold structures
  - (e) Cross-family activation (works across unrelated families — not homology)
  - (f) Absent from all existing databases → genuinely novel biology

**Figure 6**: Predictive tasks
- (a) Variant pathogenicity prediction: ROC curves vs AlphaMissense, EVE, ESM-1v
- (b) Example interpretable prediction: "variant disrupts Feature #X (catalytic triad)"
- (c) Binding site prediction accuracy vs P2Rank
- (d) Performance stratified by homology level (SAE advantage in twilight zone)

**Supplementary**: Ablation studies, full feature catalogs, additional case studies, methods details

---

## VI. Timeline and Milestones

```
Week 1-3:   [DONE] Environment setup + initial SAE training
            ├── Layers 3,7,11,15 trained (without normalization)
            ├── Discovered and fixed: pre-TopK ReLU bug, b_dec normalization bug
            └── Implemented sqrt(d) normalization + auxk ReLU fix

Week 3-5:   [IN PROGRESS] Complete SAE training with fixes
            ├── Layer 19 training (with sqrt(d) norm, k=256)
            ├── Train layers 23, 27, 31, 35
            ├── Retrain layers 3-15 with normalization (for consistency)
            └── Quality checks: FVU < 0.05, dead < 5%

            ★ Milestone 1: All 9 layers trained with consistent quality

Week 5-7:   Annotation benchmark (Phase 2)
            ├── Extract SAE features on full Swiss-Prot
            ├── Build feature→annotation classifiers
            ├── Implement all baselines (InterProScan, BLAST, linear probes, k-NN)
            ├── Held-out evaluation with stratification by homology
            └── Temporal validation (Swiss-Prot version history)

            ★ Milestone 2: Benchmark complete — SAE annotation performance quantified

Week 7-9:   Cross-layer information hierarchy (Phase 3)
            ├── Train linear probes at all 9 layers for 8 biological properties
            ├── Layer × annotation heatmap from Phase 2 SAE features
            ├── Feature granularity analysis (activation span vs depth)
            └── Practical layer selection guide

Week 9-11:  Dark proteome annotation (Phase 4)
            ├── Annotation transfer to TrEMBL/DUF/dark kinases/orphan GPCRs
            ├── Temporal validation
            ├── Case studies on dark kinases and DUF families
            └── Coverage and confidence analysis

            ★ Milestone 3: Dark proteome annotations generated

Week 10-13: Predictive tasks + novel feature validation (Phases 5-6)
            ├── Variant pathogenicity prediction (vs AlphaMissense, EVE)
            ├── Binding site prediction (vs P2Rank)
            ├── Novel feature validation pipeline (with all controls)
            ├── Select 3-5 novel features for deep case studies
            └── Deep case studies (coevolution, structure, literature)

            ★ Milestone 4: Predictive benchmarks + validated novel features

Week 13-16: Ablation studies + paper writing (Phase 7)
            ├── Expansion factor / k / layer / model scale ablations
            ├── ESM-2-650M comparison (reuse InterProt SAE)
            ├── Paper writing
            ├── Figure generation
            └── Supplementary materials

            ★ Milestone 5: Paper complete
```

---

## VII. Expected Outcomes

### 7.1 Quantitative Expectations

| Outcome | Expected | Confidence |
|---------|----------|------------|
| Annotation benchmark: SAE outperforms BLAST on orphan proteins (<30% identity) | Yes | High |
| Annotation benchmark: SAE competitive with InterProScan on well-characterized proteins | Comparable or slightly below | Medium |
| Layer hierarchy: secondary structure peaks at mid-layers (11-19) | Yes | High |
| Layer hierarchy: functional features concentrated in late layers (27-35) | Yes | High |
| Dark proteome: >50% of dark kinases gain at least one feature annotation | Yes | Medium |
| DUF families: functional hypotheses for ≥10 DUF families | Yes | Medium |
| Variant prediction: competitive with EVE (within 5% AUROC) | Likely | Medium |
| Variant prediction: interpretable explanations for >80% of predictions | Yes | High |
| Novel features: 10-50 passing multi-level validation | 15-30 | Medium |

### 7.2 Minimum Guaranteed Outcomes

Even if some hypotheses fail:
- First **rigorous annotation benchmark** for protein SAE features
- First **cross-layer biological information hierarchy** characterization via SAE features in a protein language model
- First **dark proteome application** of SAE-based annotation
- First **interpretable variant effect predictor** based on SAE features
- Comprehensive ablation study (expansion, k, depth, scale, architecture)

---

## VIII. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SAE annotation doesn't beat linear probes | Medium | High | Focus on interpretability advantage even if accuracy is similar; show SAE provides *explanations* that probes cannot |
| Layer hierarchy analysis shows no clear pattern | Low | Low | Linear probing is well-established; biological hierarchy in PLMs is widely expected |
| No convincing novel features | Low | Medium | Paper still stands on benchmark + circuits + dark proteome |
| Compute insufficient for all 9 layers | Low | Low | Prioritize 5 layers: [3, 11, 19, 27, 35] |
| Competition publishes similar work | Medium | High | Differentiators: benchmark + circuits (no competitor does both) |

---

## IX. Key References

### Core Prior Work
1. Simon & Zou. "InterPLM." *Nature Methods* (2025).
2. Adams et al. "InterProt." *ICML 2025 Spotlight*.
3. Gujral et al. "SAEs on PLMs." *PNAS* (2025).
4. Parsan et al. "SAEs for protein structure." *arXiv* (2025).

### Anthropic Interpretability (Methodology)
5. Bricken et al. "Towards Monosemanticity." *Anthropic* (2023).
6. Templeton et al. "Scaling Monosemanticity." *Anthropic* (2024).
7. Lindsey et al. "On the Biology of a Large Language Model." *Anthropic* (2025).
8. Lindsey et al. "Circuit Tracing." *Anthropic* (2025).

### Protein Models
9. Lin et al. "ESM-2." *Science* (2023).

### Annotation & Prediction Benchmarks
10. Radivojac et al. "CAFA." *Nature Methods* (2013).
11. Cheng et al. "AlphaMissense." *Science* (2023).
12. Frazer et al. "EVE." *Nature* (2021).

### Databases
13. UniProt Consortium. "UniProt 2025." *NAR* (2025).
14. Paysan-Lafosse et al. "InterPro 2025." *NAR* (2025).
15. Landrum et al. "ClinVar." *NAR* (2024).
16. Chen et al. "gnomAD v4." *Nature* (2024).
