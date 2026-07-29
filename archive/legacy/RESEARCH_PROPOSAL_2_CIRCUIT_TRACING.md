# Research Proposal 2: Circuit Tracing in Protein Generative Models

## Mechanistic Interpretability of Conditional Protein Generation via Sparse Autoencoders

**Date**: 2026-03-24
**Target venue**: Nature Machine Intelligence (primary), ICML/NeurIPS (alternative)
**Relationship**: Companion paper to Research 1 (ESM-2 encoder annotation)

---

## I. Executive Summary

### The Idea

Apply Anthropic-style circuit tracing with Sparse Autoencoders (SAEs) to **ProGen2** — a 6.4B-parameter decoder-only protein generative model — to mechanistically understand how the model generates proteins with specific functions. When conditioned to "generate a kinase," what internal circuits activate? How do features representing catalytic motifs, regulatory loops, and fold topology compose during autoregressive generation?

### Honest Assessment: Is This a Good Idea?

**VERDICT: Yes, with caveats.**

**Strengths:**
- **Genuinely novel**: Zero published papers on SAEs applied to any decoder-only protein generative model. Zero circuit tracing studies on any biological generative model.
- **Methodologically appropriate**: Circuit tracing is designed for decoder-only autoregressive models (unlike ESM-2 encoder, where we correctly dropped this approach)
- **High-impact question**: Understanding how generative protein models work is critical for trustworthy protein design in therapeutics
- **ProGen2 is experimentally validated**: Generated lysozymes were shown to be catalytically active (Nature Biotechnology 2023)

**Risks:**
- **Uninterpretable circuits** (HIGH): You may trace circuits that don't yield human-understandable biological insight
- **No clear evaluation metric** (HIGH): Unlike annotation F1, there's no ground truth for "correct circuit"
- **Competition from IBM**: An NeurIPS AI4Mat 2025 workshop paper trained SAEs on a 289M chemistry model (SMI-TED). Not on proteins, not circuit tracing, but establishes the direction
- **Feature steering validation is hard**: Even if you find and steer features, validating the generated proteins requires wet-lab work or molecular simulation

**Strategic recommendation**: Pursue as Paper 2 AFTER ESM-2 annotation paper (Research 1) establishes credibility. The two papers together form "encoder interpretability + decoder interpretability" — a compelling research program.

---

## II. Why ProGen2, Not a Drug/Small-Molecule Model

### Models Evaluated

| Model | Type | Params | Conditional? | Suitability |
|-------|------|--------|-------------|-------------|
| **ProGen2-xlarge** | Protein, decoder-only | 6.4B | Via fine-tuning | **A+** — Best choice |
| **ProGen v1** | Protein, decoder-only | 1.2B | Yes (family tags) | **A** — Built-in conditioning |
| **ProGen2-large** | Protein, decoder-only | 2.7B | Via fine-tuning | **A** — Sweet spot for compute |
| ProtGPT2 | Protein, decoder-only | 738M | No | B- — Unconditional limits circuit Q's |
| SAFE-GPT | Small molecule, decoder | 87M | Fragment-based | B+ — Best drug model option |
| cMolGPT | Small molecule, decoder | Very small | Target-specific | C+ — Too small |
| ChemGPT | Small molecule, decoder | ~1B | No | N/A — Weights unavailable |
| Token-Mol | Small molecule, decoder | ~100M | Property-based | B+ — Unified representation |

### Why Protein > Small Molecule for Circuit Tracing

1. **Token semantics**: Amino acid tokens are well-defined chemical entities with known properties. SMILES tokens (C, c, N, (, ), 1, 2) are often syntactic bookkeeping — not meaningful intervention points for circuit tracing.

2. **Model size**: ProGen2-xlarge (6.4B) is ~70× larger than SAFE-GPT (87M). Larger models develop richer internal representations → more interesting SAE features and circuits.

3. **Biological interpretability**: Protein sequence positions map directly to 3D structure. "Feature X activates at positions 45-52" can be mapped to a specific helix or binding loop. SMILES positions have no such spatial correspondence.

4. **Experimental validation**: ProGen-generated proteins have been experimentally tested (Nature Biotechnology). Generated drug molecules from SAFE-GPT have not.

5. **Conditional generation story**: Fine-tuning ProGen2 on specific protein families creates a clean experimental setup: "What circuits differ between kinase generation and GPCR generation?"

---

## III. Research Design

### 3.1 Phase 1: SAE Training on ProGen2 (Weeks 1-4)

**Model selection**: ProGen2-large (2.7B) as primary, with ProGen2-xlarge (6.4B) scaling experiment.

**Why 2.7B as primary**:
- Fits on single L20 GPU for inference (~5.4 GB FP16)
- 32 layers, d_model=2560, 32 heads
- Train SAEs at 8 layers: [3, 7, 11, 15, 19, 23, 27, 31]
- Can train multiple layers in parallel across 8 GPUs

**SAE architecture**: Same as Research 1 (TopK, sqrt(d) normalization, auxk loss)
- d_sae = 16,384 (6.4× expansion)
- k = 256
- 500K training steps per layer

**Training data**: Generate protein sequences autoregressively from ProGen2, extract intermediate activations during generation. This captures the model's actual generative computation, not just static representations.

**Key difference from encoder SAEs**:
- Encoder SAEs: train on hidden states of existing proteins (static analysis)
- Decoder SAEs: train on hidden states DURING generation (dynamic computation)
- The features we find will represent generative "decisions," not just structural properties

### 3.2 Phase 2: Conditional Generation Experiments (Weeks 4-7)

**Setup**: Fine-tune ProGen2 on family-specific protein sets, then compare SAE feature activations during conditional vs. unconditional generation.

**Experiment 1: Family-conditioned generation**
```
Conditions to compare:
  - Unconditional (baseline): generate any protein
  - Kinase family: fine-tune on ~500 kinases from Pfam
  - GPCR family: fine-tune on ~800 GPCRs
  - Lysozyme family: fine-tune on ~300 lysozymes (experimentally validated)
  - Globin family: fine-tune on ~1000 globins

For each condition:
  1. Generate 10,000 proteins
  2. Record SAE feature activations at every layer, every position
  3. Identify "condition-specific features": features that activate significantly
     more under condition X than unconditional baseline
  4. Map these features to known biology (catalytic sites, fold elements, etc.)
```

**Experiment 2: Feature steering**
```
Hypothesis: Amplifying condition-specific features during generation
should produce more proteins with those properties.

Protocol:
  1. Identify top-50 kinase-specific SAE features
  2. During unconditional generation, artificially amplify these features
  3. Evaluate generated sequences:
     - Fold prediction (ESMFold) — do they fold into kinase-like structures?
     - Function prediction (InterProScan) — are they classified as kinases?
     - Sequence identity to known kinases — are they truly novel?
  4. Compare steering quality to standard fine-tuning
```

### 3.3 Phase 3: Circuit Tracing (Weeks 6-10)

**This is the core contribution and where decoder-only architecture enables true causal analysis.**

**Method: Attribution Patching (Gradient-Based)**

For a target feature f_target at layer L_late and a candidate upstream feature f_upstream at layer L_early:

```python
# Pseudocode for attribution patching
def compute_attribution(model, sae_early, sae_late, protein_batch):
    # Forward pass with hooks
    activations = model.forward_with_hooks(protein_batch)

    # Get SAE features at both layers
    f_early = sae_early.encode(activations[L_early])  # (batch, seq, d_sae)
    f_late = sae_late.encode(activations[L_late])      # (batch, seq, d_sae)

    # For each target feature at L_late:
    target_activation = f_late[:, :, target_idx]

    # Compute gradient of target w.r.t. upstream features
    # (one backward pass per target feature)
    grad = torch.autograd.grad(target_activation.sum(), f_early)

    # Attribution = gradient × activation (integrated gradients style)
    attribution = (grad * f_early).sum(dim=(0, 1))  # (d_sae,)

    return attribution  # which upstream features most influence the target
```

**Computational tractability** (co-activation filtering):
1. Build co-activation graph first (~2 GPU-hours): for each feature pair across layers, count how often both are active on the same protein
2. Only compute attributions for pairs with co-activation > 50 proteins
3. This reduces search space by ~1000×
4. Validate top-50 circuit edges with actual activation patching (ablation)

**Target circuits to discover:**

| Circuit | Early-layer features | Late-layer features | Biological question |
|---------|---------------------|--------------------|--------------------|
| **Hydrophobic core packing** | Hydrophobic AA preferences (L3-7) | Core packing features (L19-27) | How does the model build a hydrophobic core during generation? |
| **Catalytic site assembly** | Active site residue features (L7-15) | Catalytic triad/site features (L23-31) | How are catalytic residues coordinated across sequence during generation? |
| **Secondary structure planning** | Local propensity features (L3-11) | Helix/sheet continuation features (L15-23) | Does the model "plan ahead" for secondary structure? |
| **Family identity** | Conserved motif features (L7-15) | Family-level features (L27-31) | Where is "kinase-ness" or "GPCR-ness" represented? |

### 3.4 Phase 4: Biological Validation (Weeks 9-13)

**Validate discovered circuits against known protein biology:**

1. **Catalytic site circuit**: Map circuit features to known catalytic mechanisms (Catalytic Site Atlas). Do the features correspond to known catalytic residue types?

2. **Fold topology circuit**: Compare feature activation patterns to CATH/SCOP fold classifications. Do late-layer features correspond to fold families?

3. **Steering validation**: Generate proteins by steering specific circuit components. Submit to ESMFold for structure prediction. Do steered proteins have the expected 3D fold?

4. **Cross-family analysis**: Run the same proteins through both ESM-2 SAEs (Research 1) and ProGen2 SAEs (Research 2). Do the encoder and decoder represent the same biology? Or do they capture complementary aspects?

### 3.5 Phase 5: Scaling and Ablation (Weeks 11-14)

| Experiment | Purpose |
|-----------|---------|
| ProGen2-small (151M) vs medium (764M) vs large (2.7B) vs xlarge (6.4B) | How does model scale affect circuit complexity? |
| SAE expansion factor (8K vs 16K vs 32K) | How many features are needed to capture generative circuits? |
| Layer depth analysis | Which layers are most important for conditional generation? |
| Comparison with attention analysis | Do SAE circuits reveal more than attention head analysis? |

---

## IV. Compute Plan

| Task | GPUs | Time |
|------|------|------|
| SAE training on ProGen2-2.7B (8 layers) | 8 × L20 | ~2-3 weeks |
| SAE training on ProGen2-6.4B (4 layers) | 8 × L20 | ~2-3 weeks |
| Conditional generation + feature analysis | 2 × L20 | ~1 week |
| Circuit tracing (attribution patching) | 4 × L20 | ~1 week |
| Steering experiments + validation | 2 × L20 | ~1 week |
| **Total** | | **~7-9 weeks** |

**Can overlap with Research 1**: SAE training for Research 2 can run on GPUs not being used for Research 1's remaining layers.

---

## V. Paper Structure

### Title Options
1. "Circuit Tracing in Protein Generative Models: How ProGen2 Builds Functional Proteins"
2. "Mechanistic Interpretability of Conditional Protein Generation"
3. "From Features to Circuits: Understanding How Language Models Generate Proteins"

### Figures

**Figure 1**: Method overview
- (a) ProGen2 architecture + SAE attachment points
- (b) Conditional generation setup (family-specific fine-tuning)
- (c) Circuit tracing methodology schematic

**Figure 2**: SAE features in a generative protein model
- (a) Feature catalog: amino acid preference features, structural features, family features
- (b) Feature activation patterns during generation of a kinase vs. GPCR
- (c) Condition-specific features: which features activate only under kinase conditioning?

**Figure 3**: Circuit analysis (THE KEY FIGURE)
- (a) Attribution graph: how catalytic site features compose across layers
- (b) Attribution graph: how family identity propagates from early to late layers
- (c) Circuit complexity vs. model scale (2.7B vs 6.4B)

**Figure 4**: Feature steering
- (a) Steering kinase features during unconditional generation → kinase-like outputs
- (b) ESMFold structures of steered vs. unsteered proteins
- (c) InterProScan annotation of steered proteins
- (d) Comparison: steering quality vs. fine-tuning quality

**Figure 5**: Encoder vs. Decoder comparison
- (a) Same proteins analyzed by ESM-2 SAEs (encoder, Research 1) vs. ProGen2 SAEs (decoder)
- (b) Shared features (both models capture) vs. unique features
- (c) "Understanding" vs. "generating": complementary aspects of protein biology

---

## VI. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Circuits are uninterpretable | Medium-High | High | Focus on well-understood biology (catalytic sites); publish feature analysis even if circuits are noisy |
| ProGen2 features are trivial (just AA preferences) | Medium | High | Use 6.4B model; compare against random baselines |
| Steering doesn't work cleanly | Medium | Medium | Paper can focus on circuit analysis without steering |
| No clear evaluation metric for circuits | High | Medium | Use biological ground truth (catalytic mechanisms) as validation |
| IBM follow-up paper takes novelty | Low | Medium | Our work is on proteins (not small molecules) and includes circuit tracing (not just features) |

---

## VII. Timeline Integration with Research 1

```
Research 1 (ESM-2 Annotation):
  Weeks 1-16: SAE training → Annotation benchmark → Dark proteome → Paper

Research 2 (ProGen2 Circuits):
  Weeks 8-22: Start after Research 1 SAE training completes
  ├── Weeks 8-11: SAE training on ProGen2
  ├── Weeks 11-14: Conditional generation + circuit tracing
  ├── Weeks 14-17: Validation + scaling experiments
  └── Weeks 17-22: Paper writing

Overlap: Weeks 8-16 both projects run in parallel
  - Research 1 uses 4 GPUs (annotation benchmark, dark proteome)
  - Research 2 uses 4 GPUs (SAE training on ProGen2)
```

---

## VIII. Key References

1. Madani et al. "Large language models generate functional protein sequences across diverse families." *Nature Biotechnology* (2023). [ProGen]
2. Nijkamp et al. "ProGen2: Exploring the boundaries of protein language models." *Cell Systems* (2023).
3. Lindsey et al. "Circuit Tracing." *Anthropic* (2025).
4. Lindsey et al. "On the Biology of a Large Language Model." *Anthropic* (2025).
5. IBM. "Unveiling Latent Knowledge in Chemistry Language Models through Sparse Autoencoders." *NeurIPS AI4Mat Workshop* (2025).
6. Ferruz et al. "ProtGPT2." *Nature Communications* (2022).
7. Krenn et al. "SELFIES." *Machine Learning: Science and Technology* (2020).
8. Noutahi et al. "Gotta be SAFE: A new framework for molecular design." *Digital Discovery* (2024). [SAFE-GPT]
