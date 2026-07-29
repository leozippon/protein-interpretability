# R1 Problem Analysis for Opus

Prepared after the 2026-05-16 Opus-addition execution pass.

## Short Diagnosis

R1 is not uniformly bad. The current evidence supports a narrower statement:
the ESM-2 SAE features contain real pathogenicity and annotation signal, but
they do not support the originally hoped-for claims that SAE is a stronger
clinical pathogenicity predictor, a robust LOF/GOF/DN mechanism classifier, or
a systematic AlphaMissense blind-spot corrector.

I do not currently think the main failure mode is "the SAE training is broken."
The stronger diagnosis is that the paper hypothesis has outrun what an
unsupervised SAE over ESM-2 hidden states can reliably provide for clinical
variant mechanism prediction.

## Evidence That the SAE Is Not Simply Broken

The trained R1 SAEs show reasonable basic health:

- Architecture: ESM-2-3B layers 19, 23, 27, 31, 35; TopK SAE with
  `d_sae=16384`, `k=256`; trained 500k steps.
- Alive feature rate is high in the final annotation summaries:
  - L19: 15,944 / 16,384 alive, 97.3%.
  - L23: 15,991 / 16,384 alive, 97.6%.
  - L27: 15,908 / 16,384 alive, 97.1%.
  - L31: 15,928 / 16,384 alive, 97.2%.
  - L35: 15,186 / 16,384 alive, 92.7%.
- Annotation alignment is non-trivial after the firing-position rerun:
  - L19: 381 KNOWN, 3,483 PARTIAL, 7,481 useful features.
  - L23: 301 KNOWN, 2,992 PARTIAL, 6,773 useful features.
  - L27: 163 KNOWN, 1,725 PARTIAL, 4,423 useful features.
  - L31: 86 KNOWN, 1,135 PARTIAL, 3,197 useful features.
  - L35: 45 KNOWN, 653 PARTIAL, 1,935 useful features.
- Pathogenicity signal exists:
  - ClinVar2000 SAE-LR AUC: 0.8782.
  - ESM-2 LLR AUC: 0.8822.
  - SAE+LLR AUC: 0.9143.
  - CancerHoldout101 SAE-LR AUC: 0.9079.
  - CancerHoldout101 SAE+LLR AUC: 0.9193.

If the SAE were fundamentally broken, I would expect dead-feature collapse,
near-random annotation alignment, and near-random pathogenicity AUC. We do not
see that pattern.

## Evidence That the Strong R1 Claims Fail

### 1. Scalar pathogenicity is not competitive with specialist predictors

Accessible external baselines are stronger than SAE+LLR on raw missense
pathogenicity:

- ClinVar2000:
  - SAE-LR: 0.8782.
  - ESM-2 LLR: 0.8822.
  - SAE+LLR: 0.9143.
  - AlphaMissense: 0.9474.
  - gMVP: 0.9369.
  - ESM-1v: 0.9089.
- CancerHoldout101:
  - SAE-LR: 0.9079.
  - ESM-2 LLR: 0.8978.
  - SAE+LLR: 0.9193.
  - AlphaMissense: 0.9700.
  - gMVP: 0.9400.
  - ESM-1v: 0.8552.

The 2026-05-11 AlphaMissense ensemble gate was also negative:

- AlphaMissense AUC: 0.9474.
- AM+SAE stacked AUC: 0.8542.
- AM+SAE z-sum AUC: 0.9210.
- Both deltas versus AlphaMissense were negative under group bootstrap.

Interpretation: SAE cannot be framed as an AlphaMissense-improving scalar
pathogenicity method in the current evidence package.

### 2. LOF/GOF/DN mechanism generalization fails

The mechanism classifier has signal under easier splits but fails under the
protein-level split:

- Variant-CV:
  - SAE-only macro-AUC: 0.7471.
  - LLR-only macro-AUC: 0.5054.
  - SAE+LLR macro-AUC: 0.7488.
- Protein-level split:
  - SAE-only macro-AUC: 0.5161.
  - LLR-only macro-AUC: 0.4642.
  - SAE+LLR macro-AUC: 0.5164.
- Gene-level mechanism proxy:
  - Macro-AUC: 0.5665.
  - Macro-F1: 0.3769.
- Channelopathy expert-label concordance:
  - Accuracy: 0.625.
  - Macro-F1: 0.444.
  - Dominant-negative cases mostly collapse into LOF-like predictions.

Interpretation: the SAE captures variant-local and protein-specific mechanism
structure, but the learned mechanism head does not transfer across proteins
well enough for a general mechanism-prediction claim.

### 3. ProteinGym is a negative diagnostic

ProteinGym / DMS results do not support SAE as a generic fitness-effect scorer:

- ProteinGym assays scored: 217 / 217.
- LLR average: 0.4341.
- SAE average: -0.2314.
- SAE+LLR z-sum: 0.1993.
- Sign-corrected SAE+LLR: 0.4047, still below LLR.
- Ensemble-minus-LLR mean delta: -0.2348.

Interpretation: SAE perturbation magnitude is not aligned with generic DMS
fitness effects. This is not necessarily a training failure; clinical
pathogenicity and assay fitness are related but not identical targets.

### 4. Indel extension is useful but not a standalone SAE win

The completed R1-Add-1 indel protein-sequence baseline gives:

- SAE+ESM+cheap grouped LR: 0.9108.
- Cheap grouped LR: 0.8447.
- ESM region mean pseudo-NLL: 0.8037.
- SAE damage score: 0.7735.
- Early truncation score: 0.7508.

Interpretation: SAE contributes to a strong combined model, but SAE damage
alone is not the strongest protein-coordinate-free indel baseline. The current
claim should be "IndelMissense v1 plus a strong combined interpretable baseline"
rather than "SAE is a superior indel scorer."

### 5. AM-vs-SAE residual typing is negative

The 2026-05-16 disagreement typing found:

- Opposite-direction AM/SAE disagreements: 473.
- Review-filtered disagreements: 321.
- AM right / SAE wrong: 283.
- SAE right / AM wrong: 38.
- BH-significant context enrichments: 0.

Interpretation: the curated residual cases remain useful examples, but there is
no statistical basis yet for a systematic "SAE fixes AlphaMissense blind spot X"
claim.

## Most Likely Failure Modes

### A. Claim-target mismatch

The SAE is an unsupervised decomposition of ESM-2 hidden states. AlphaMissense
and gMVP are supervised or resource-integrated specialist predictors. It is not
surprising that an SAE interpretation layer loses to these models on scalar
pathogenicity.

### B. Mechanism labels are not residue-local enough

LOF/GOF/DN labels often depend on disease mechanism, protein complex context,
expression, dominance, pathway role, gain-of-function pharmacology, and
clinical curation. A local ESM hidden-state perturbation can capture structural
or functional disruption, but may not identify whether that disruption maps to
LOF, GOF, or dominant-negative biology across genes.

### C. Strong protein identity leakage in easy splits

Variant-CV mechanism AUC around 0.75 but protein-level AUC around 0.52 suggests
that the original mechanism signal partly used protein/gene-specific context.
This is scientifically informative but not a generalizable mechanism classifier.

### D. Annotation-guided SAE features are useful but incomplete

The annotation alignment is strongest at L19/L23 and weaker in deeper layers.
This may be enough for interpretable case studies and pathogenicity support,
but not enough for a stable global mechanism taxonomy.

### E. Current aggregation may discard important structure

Several R1 scorers collapse feature perturbations into scalar summaries or
annotation-weighted sums. The useful information may be high-dimensional but
sample-limited. The full 163k-dimensional vector LR overfits on 2,000 variants,
while annotation-selected features recover AUC 0.878. This suggests that the
representation has information, but the current supervised sample size and
regularization are limiting.

## Is SAE Training Itself Still a Suspect?

It remains a secondary suspect, but not the leading one.

Reasons to keep it on the list:

- We need a clean reconstruction-quality table for the final R1 layers in the
  same format used for R2 CLTs. The annotation summaries show alive rates but
  not a compact per-layer FVU table in the current status document.
- Deep layers show weaker annotation alignment, especially L35.
- The current TopK `k=256` representation may be too dense for mechanism
  interpretability or too lossy for variant-local perturbations, depending on
  the layer.

Reasons it is probably not the main blocker:

- Alive rates are high.
- Annotation alignment exists.
- SAE-LR pathogenicity AUC is high enough to show meaningful signal.
- The biggest failures occur exactly where the target label is hardest and
  least directly encoded: cross-protein LOF/GOF/DN, AlphaMissense residual
  correction, and ProteinGym generic DMS fitness.

## Critical Diagnostic Experiments to Resolve the Ambiguity

These are designed to distinguish "SAE training/representation is weak" from
"the scientific target is not encoded in ESM local perturbations."

### Diagnostic 1: Raw-hidden vs SAE-code vs SAE-reconstruction perturbation

Run the same grouped split for three feature sources:

1. Raw ESM hidden-state perturbation at layers 19/23/27/31/35.
2. SAE reconstruction-space perturbation.
3. SAE code-space perturbation.

Run for:

- ClinVar pathogenicity.
- LOF/GOF/DN mechanism.
- IndelMissense.

Interpretation:

- If raw hidden >> SAE code, then the SAE may be losing task-relevant
  information.
- If raw hidden also fails on mechanism, then the mechanism claim is likely
  target-limited rather than SAE-limited.
- If SAE reconstruction matches raw hidden but code fails, the sparse code is
  too lossy or the aggregation is wrong.

### Diagnostic 2: Protein-identity leakage audit

Quantify how much mechanism performance comes from protein identity:

- Gene-only baseline.
- Protein-family-only baseline.
- ESM sequence embedding without variant perturbation.
- Variant perturbation after regressing out protein identity / family.

Interpretation:

- If gene/family baseline is strong under variant-CV and collapses under
  protein-CV, the original mechanism success is largely protein identity.
- If perturbation still adds signal after identity control, R1 can retain a
  narrower mechanism-diagnostic claim.

### Diagnostic 3: Per-gene/per-class failure map

For LOF/GOF/DN and channelopathy:

- Report confusion by gene and class.
- Separate DN-vs-LOF specifically.
- Identify whether failures cluster in a few genes or are globally unstable.

Interpretation:

- A few-gene failure suggests data/label issue.
- Global DN collapse suggests the representation cannot distinguish dominant
  negative biology from loss-of-function disruption.

### Diagnostic 4: Layer ablation for R1

Repeat pathogenicity and mechanism probes using each SAE layer alone and
selected layer groups:

- L19 only.
- L23 only.
- L19+L23.
- L27+L31+L35.
- All layers.

Interpretation:

- If L19/L23 dominate, deeper layers may add noise.
- If deeper layers help mechanism but hurt pathogenicity, split the paper
  claims by task.

### Diagnostic 5: Label-quality stratification

For mechanism and ClinVar labels:

- Stratify by ClinVar review stars.
- Stratify by manually curated versus inferred mechanism labels.
- Stratify by disease/gene family.

Interpretation:

- If performance improves sharply on high-confidence labels, the bottleneck is
  label noise.
- If it remains poor, the target is likely not represented well enough.

## Recommended Reframing If No New Positive Result Appears

The strongest defensible R1 paper is not "SAE beats clinical predictors." It is:

1. A rigorous audit of what unsupervised protein-LM SAEs can and cannot do for
   variant interpretation.
2. A useful interpretable perturbation layer that matches ESM-2 LLR on
   ClinVar-scale pathogenicity and improves LLR when combined.
3. A feature-level diagnostic resource with annotated firing positions and
   case studies.
4. IndelMissense v1 as a bounded, reusable protein-sequence indel benchmark,
   with baseline tables showing where SAE helps and where simple protein
   baselines dominate.
5. A negative but important finding: SAE interpretability does not automatically
   yield cross-protein clinical mechanism prediction.

This is more likely to be publishable as a careful methods/resource paper than
as a top-tier "new clinical predictor" paper.

## Questions for Opus

1. Should R1 be reframed now as an audit/resource paper, or should we run the
   raw-hidden vs SAE-code diagnostic first before making the final decision?
2. Is the positive SAE+LLR ClinVar result strong enough to remain a central
   contribution, given that AlphaMissense/gMVP are stronger?
3. Should IndelMissense v1 be promoted as a standalone resource even though SAE
   damage alone is not the strongest baseline?
4. Should LOF/GOF/DN be removed from the main narrative and moved to a negative
   diagnostic appendix?
5. What minimum new positive result would be required to keep R1 at a
   Nature Methods target rather than downgrading to Bioinformatics / Genome
   Biology / NAR-style resource framing?
