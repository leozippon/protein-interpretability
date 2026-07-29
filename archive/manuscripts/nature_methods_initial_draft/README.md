# Nature Methods Initial Draft

This directory contains a first BioCC manuscript scaffold based on the official
Springer Nature LaTeX authoring template, configured with the `sn-nature` class
option for Nature Portfolio journals.

Official template sources retained here:

- `official_template/springer_nature_latex_template_dec2024.zip`
- `official_template/extracted/sn-article-template/`

Why this template:

- Nature Methods' TeX/LaTeX formatting page points authors to Springer Nature's
  LaTeX author support page for the template package.
- Springer Nature's author support page states that the journal article template
  can be downloaded as a zip or accessed through Overleaf.
- The Overleaf gallery lists the Springer Nature LaTeX Template as the official
  Springer Nature authoring template and includes the `sn-nature` class option
  for Nature Portfolio journals.

Draft files:

- `main.tex`: initial Nature Methods-style manuscript filled with current R1/R2
  methods, results, limitations and TODOs.
- `references.bib`: minimal working bibliography for compilation.
- `sn-jnl.cls` and `bst/`: copied from the official template package so the
  draft can compile locally and be uploaded to Overleaf.

Compile locally:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The local TeX Live installation needed the following official CTAN packages for
the Springer Nature class to compile: `sttools`, `threeparttable`, `appendix`,
`wrapfig` and `multirow`.

Notes:

- This is not submission-ready. It intentionally includes current negative
  results and TODO markers for missing competitors, wet-lab validation,
  finalized authorship and data/code availability.
- Nature Methods guidance allows TeX/LaTeX submissions and points authors to the
  Springer Nature LaTeX template. For final submission, check journal-level
  instructions again and paste the compiled bibliography into the `.tex` file if
  required by the submission system.
