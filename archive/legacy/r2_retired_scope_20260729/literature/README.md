# Literature corpus for Paper A

This directory contains an open-access reading corpus assembled for the
2026-07-16 revision of Paper A. The primary venue is **npj Artificial
Intelligence**; **npj Systems Biology and Applications** is the topical backup.

## Corpus and selection rule

- 20 official Nature-hosted open-access PDFs.
- 10 papers from `npj Artificial Intelligence` and 10 from `npj Systems
  Biology and Applications`.
- Publication window: 2023-07-16 through 2026-07-16.
- Selection themes: explanation validity, robustness and null controls, sparse
  internal representations, causal testing, molecular or protein language
  models, enzyme prediction and biologically grounded validation.
- Every PDF was validated with `pdfinfo`; filenames, dates, DOI links and
  local paths are recorded in `npj_recent/MANIFEST.tsv`.

The corpus intentionally includes both close precedents and contrast cases.
Not every downloaded paper should be cited in the manuscript.

## Main synthesis for the manuscript

The recent npj literature supports a three-part distinction that should govern
Paper A's claims:

1. **Reproducibility is not semantic validity.** Stable or recurrent features
   can still reflect architecture, local sequence statistics, position or
   other nuisance structure. Esser-Skala and Fortelny show that biological
   interpretations can vary across repeated fits and inherit network biases.
2. **Semantic alignment is not faithfulness.** Lee and colleagues explicitly
   separate human-aligned explanations from faithful participation in the
   model's decision. Paper A should use independent labels and interventions
   rather than treating interpretable examples as sufficient evidence.
3. **Faithfulness is not real-world causal control.** Haufe and colleagues
   argue that explanation questions require formal correctness criteria, and
   Pesapane and colleagues prioritize prospective evidence over plausible
   explanations. Paper A's permutation controls, matched interventions and
   negative gates fit this direction, provided their limitations are stated.

For protein applications, KinForm highlights family-aware evaluation and the
risk that apparently strong functional prediction can reflect family identity.
The interpretable enzyme-class KAN paper offers a neighboring example of motif
validation, while the antibody and molecular-language-model papers establish
current biological foundation-model context. These precedents do not validate
Paper A's features; they clarify which external tests would be needed.

## Priority reading

### Tier 1 — direct conceptual anchors

- **Explainable AI needs formalization** — strongest justification for
  separating the question being asked from the explanation algorithm and for
  defining correctness criteria before interpretation.
- **Reliable interpretability of biology-inspired deep neural networks** —
  closest biological precedent for replicated training, control inputs and
  bias-aware interpretation.
- **Evidence over explanations: put medical AI to the test** — concise support
  for prospective tests, invariance, negative controls and bounded claims.
- **Toward faithful and human-aligned self-explanation of deep models** — useful
  distinction between predictive performance, faithfulness, stability and
  human alignment.
- **How large language models encode theory-of-mind: a study on sparse
  parameter patterns** — a positive sparse-pattern intervention precedent that
  helps frame Paper A's negative intervention results.

### Tier 2 — protein and biological validation anchors

- **KinForm** — family-aware splits, intermediate protein-language-model
  representations and enzyme-property generalization.
- **Interpretable Kolmogorov-Arnold networks for enzyme commission number
  prediction** — input-level enzyme interpretation validated against motifs;
  useful as a contrast to internal-feature and causal-control claims.
- **Context-aware multi-property antibody predictor** — current text and
  protein language model integration in a biological prediction setting.
- **Causality-aware graph neural networks for functional stratification and
  phenotype prediction at scale** — a biological example that makes causal
  structure an explicit modeling target.

### Tier 3 — broader context and contrasts

The remaining papers cover sparse connectivity, functional decomposition,
causal metric calibration, intrinsic tabular interpretability, omics XAI,
protein-function communication and generative biological design. They are
useful for methods comparison or venue style, but are less direct support for
the paper's central sparse-readout claims.

## Venue fit

`npj Artificial Intelligence` is the stronger target because Paper A's main
contribution is a general validation framework: cross-model invariance,
semantic interpretation and causal control are separate evidential questions.
The protein generators are a demanding test domain rather than the sole source
of novelty. `npj Systems Biology and Applications` remains a backup if the
manuscript is reframed around biological representation analysis.

## Reproducibility

The PDFs are unmodified publisher files. Re-run checks with:

```bash
find r2_interpretability_transfer/literature/npj_recent -name '*.pdf' -print0 | \
  xargs -0 -n1 pdfinfo >/dev/null
sha256sum -c r2_interpretability_transfer/literature/npj_recent/SHA256SUMS
```

The files are provided for research reading under their respective open-access
licenses. Consult each article PDF for the exact licence and attribution terms.
