# Recoverability probes (Experiment 45)

Skill = metric - chance. C = ceiling (R_raw), F = floor (R_code, same layer),
gap = C-F, rho = F/C, base = composition/chance baseline, phi = C/ESM2.

| Model | Task | metric | C | F | gap | rho | base | phi | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| protgpt2 | ec_topclass | macro_f1 | 0.264 | 0.270 | -0.006 | 1.000 | 0.068 | 1.00 | 280 |
| protgpt2 | pfam_family | macro_f1 | 0.939 | 0.906 | 0.033 | 0.965 | 0.855 | 1.00 | 240 |
| protgpt2 | secondary_fraction | r2 | 0.344 | -1.384 | 1.727 | 0.000 | 0.088 | 1.13 | 300 |
| protgpt2 | residue_ss | macro_f1 | 0.194 | 0.132 | 0.062 | 0.682 | 0.027 | nan | 44626 |
| protgpt2 | decoder_ec | _skipped: decoder cohort only for zymctrl_ | | | | | | | |
| zymctrl | ec_topclass | macro_f1 | 0.120 | 0.046 | 0.074 | 0.380 | 0.068 | 0.46 | 280 |
| zymctrl | pfam_family | macro_f1 | 0.898 | 0.764 | 0.134 | 0.851 | 0.855 | 0.95 | 240 |
| zymctrl | secondary_fraction | r2 | 0.172 | -26.290 | 26.462 | 0.000 | 0.088 | 0.57 | 300 |
| zymctrl | residue_ss | macro_f1 | 0.136 | 0.121 | 0.016 | 0.885 | 0.027 | nan | 44626 |
| zymctrl | decoder_ec | macro_f1 | 0.652 | 0.525 | 0.127 | 0.806 | 0.517 | nan | 48 |
| progen2-medium | ec_topclass | macro_f1 | 0.283 | 0.121 | 0.162 | 0.428 | 0.068 | 1.08 | 280 |
| progen2-medium | pfam_family | macro_f1 | 0.952 | 0.910 | 0.041 | 0.956 | 0.855 | 1.01 | 240 |
| progen2-medium | secondary_fraction | r2 | 0.169 | -14.640 | 14.809 | 0.000 | 0.088 | 0.56 | 300 |
| progen2-medium | residue_ss | macro_f1 | 0.243 | 0.198 | 0.044 | 0.817 | 0.027 | nan | 44626 |
| progen2-medium | decoder_ec | _skipped: decoder cohort only for zymctrl_ | | | | | | | |
