# Official Springer Nature LaTeX template

This directory preserves the official Springer Nature journal-article template
used for the npj Artificial Intelligence adaptation.

- Package: Springer Nature LaTeX authoring template, version 3.1, December 2024.
- Official download: <https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12>
- Author-support page: <https://www.springernature.com/gp/authors/campaigns/latex-author-support/see-where-our-services-will-take-you/18782940>
- Original archive: `springer_nature_latex_template_dec2024.zip`
- Extracted source: `springer_nature_202412/sn-article-template/`
- Integrity manifest: `SHA256SUMS`

The manuscript uses:

```tex
\documentclass[pdflatex,sn-nature]{sn-jnl}
```

The manuscript's `sn-jnl.cls` and `sn-nature.bst` are byte-for-byte identical
to the corresponding files in this official package. The journal does not
provide a separate npj Artificial Intelligence class; journal-specific Article
requirements are applied through manuscript structure, length, display-item,
statistics and availability rules.

Verify the retained package from this directory with
`sha256sum -c SHA256SUMS`.
