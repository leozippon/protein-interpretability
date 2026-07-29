# npj Artificial Intelligence adaptation audit — 2026-07-16

## Official guidance used

- Journal scope: <https://www.nature.com/npjai/aims>
- Content types: <https://www.nature.com/npjai/content-types>
- Submission guidelines:
  <https://www.nature.com/npjai/for-authors-and-referees/submission-guidelines>
- Springer Nature LaTeX support and official template:
  <https://www.springernature.com/gp/authors/campaigns/latex-author-support/see-where-our-services-will-take-you/18782940>

npj Artificial Intelligence does not distribute a journal-specific LaTeX
class. The manuscript therefore uses the official December 2024 Springer Nature
v3.1 `sn-jnl` package with the `sn-nature` option and applies the journal's
Article requirements through structure and content.

## Package mapping

| Article expectation | Implemented package |
|---|---|
| concise, non-technical abstract of about 150 words | approximately 172-word abstract after the bounded P0-7 calibration update |
| unheaded opening followed by Results, Discussion and Methods | implemented in `main.tex` |
| no more than ten main display items | six main figures; zero main tables |
| Methods and explicit statistical reporting | detailed activation-site, null, intervention, probe and statistics subsections |
| Data and Code availability | separate statements, with unresolved deposits stated rather than invented |
| supplementary information separate from the Article | standalone 12-page file with exactly eleven tables |
| Nature numbered references | `sn-nature.bst`; 43 cited references from 46 bibliography records |
| source data for plotted results | 65-row/68-checksum, sequence-safe package with deterministic builder; strict JSON normalization records every replaced historical non-finite field and Table 11 binds the synthetic calibration receipts |
| responsible AI disclosure | explicit generative-AI-tools subsection |

The introduction/results/discussion text is approximately 2,350 words after
excluding Methods and figure legends, shorter than the journal's typical
4,000–4,500-word Article range. This is not padded artificially; an editorial
presubmission check can determine whether further contextual expansion is
useful. The Article has six main display items, below the stated maximum.

## Evidence-driven changes from the previous draft

- Reframed the paper from a feature-discovery/causal-mechanism report to an
  audit separating recurrence, association and intervention.
- Removed the invalid original Swiss-Prot MI gate and added the exploratory
  matched-null repair with its selection limitations.
- Replaced raw-residual terminology with the actual architecture-specific
  normalized hook points.
- Disclosed padding in CLT training/quality diagnostics and the heuristic
  steering endpoint.
- Removed the site-mismatched direction experiment from causal conclusions.
- Recast the wider-dictionary run as exploratory because no checkpoint met the
  preregistered FVU criterion.
- Rebuilt figures to show all available intervention rows, all saved
  N-terminal contexts and machine-readable wider-dictionary values.

## Items that cannot be completed autonomously

- verified coauthor list, affiliations, contributions, funding and compute
  acknowledgements;
- licensed deposit of the recovered, hash-verified exact 200-sequence atlas cohort;
- immutable upstream model revision identifiers;
- code release tag/commit and archival DOI for data/checkpoints; and
- exact release/version metadata for staged biological databases and external
  evaluation tools where the original run did not retain it; and
- coauthor approval, ethics/licence checks and journal declarations.
