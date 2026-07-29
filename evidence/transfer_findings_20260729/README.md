# Transfer findings evidence

This package preserves the compact outputs needed to audit EXP-R2-059, EXP-R2-060, EXP-R2-062, EXP-R2-063 and EXP-R2-064 after the large local and GPFS result trees are reorganized. `artifacts/` contains byte-for-byte copies of the PAA reports and causal matrices, the eleven-arm instrument outputs and item manifests, the cohort-skip sensitivity outputs, the corrected TG outputs, and the unmasked homology-control outputs.

The files are byte-for-byte copies from the ignored `results/` trees captured on 2026-07-29 while the repository was at `5ac1f379481fe53e97fa7eb1f87fb195e35bcce5`. Verify them from this directory with `sha256sum -c SHA256SUMS`.

This receipt closes the Git durability gap for reported output values, but it cannot reconstruct provenance that the original runs never recorded. In particular, the old item manifests identify source code and commands incompletely and do not contain content digests for every model, tokenizer and corpus. Those omissions remain an accepted limitation; this package must not be represented as a fully reproducible environment snapshot.
