# R1 Nature Methods Draft

This is the R1-only manuscript split from the combined BioCC draft.

Focus:

- ESM-2 sparse autoencoder feature annotation.
- Variant perturbation diagnostics.
- Pathogenicity complementarity with LLR.
- Honest negative results for AlphaMissense/gMVP scalar competition, AlphaMissense+SAE ensembling, protein-level mechanism holdout and ProteinGym.
- Bounded indel extension over the current 6,649 binary-label reconstructable sequence edits.
- No wet-lab dependency in the current manuscript; assay validation is future work only.

Current evidence boundary:

- R1 is not a state-of-the-art scalar pathogenicity predictor.
- The defensible article is an interpretable variant-perturbation and residual-diagnostic workflow.
- PrimateAI-3D is excluded from the current accessible-baseline round because gated access was unavailable.

Compile:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The official Springer Nature class files are copied from
`../nature_methods_initial_draft/`, which retains the downloaded December 2024
template package.
