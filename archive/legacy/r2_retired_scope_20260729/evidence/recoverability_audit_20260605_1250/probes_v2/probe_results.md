# Recoverability probes (Experiment 45)

Skill = metric - chance. C = ceiling (R_raw), F = floor (R_code, same layer),
gap = C-F, rho = F/C, base = composition/chance baseline, phi = C/ESM2.

| Model | Task | metric | C | F | gap | rho | base | phi | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| protgpt2 | ec_topclass | macro_f1 | 0.260 | 0.262 | -0.002 | 1.000 | 0.051 | 0.99 | 280 |
| protgpt2 | ec_topclass_stratified | macro_f1 | 0.557 | 0.547 | 0.010 | 0.982 | 0.393 | 0.96 | 280 |
| protgpt2 | pfam_family | macro_f1 | 0.939 | 0.808 | 0.131 | 0.861 | 0.855 | 0.99 | 240 |
| protgpt2 | secondary_fraction | r2 | 0.621 | -3.155 | 3.776 | 0.000 | 0.170 | 0.90 | 300 |
| protgpt2 | residue_ss | macro_f1 | 0.221 | 0.217 | 0.004 | 0.981 | 0.027 | nan | 44626 |
| protgpt2 | decoder_ec | _skipped: decoder cohort only for zymctrl_ | | | | | | | |
| zymctrl | ec_topclass | macro_f1 | 0.123 | 0.022 | 0.101 | 0.180 | 0.051 | 0.47 | 280 |
| zymctrl | ec_topclass_stratified | macro_f1 | 0.356 | 0.229 | 0.127 | 0.643 | 0.393 | 0.61 | 280 |
| zymctrl | pfam_family | macro_f1 | 0.902 | 0.768 | 0.134 | 0.851 | 0.855 | 0.95 | 240 |
| zymctrl | secondary_fraction | r2 | 0.397 | -184.669 | 185.066 | 0.000 | 0.170 | 0.57 | 300 |
| zymctrl | residue_ss | macro_f1 | 0.149 | 0.129 | 0.020 | 0.866 | 0.027 | nan | 44626 |
| zymctrl | decoder_ec | macro_f1 | 0.652 | 0.525 | 0.127 | 0.806 | 0.517 | nan | 48 |
| progen2-medium | ec_topclass | macro_f1 | 0.272 | 0.153 | 0.119 | 0.561 | 0.051 | 1.04 | 280 |
| progen2-medium | ec_topclass_stratified | macro_f1 | 0.594 | 0.497 | 0.097 | 0.836 | 0.393 | 1.02 | 280 |
| progen2-medium | pfam_family | macro_f1 | 0.952 | 0.910 | 0.041 | 0.956 | 0.855 | 1.00 | 240 |
| progen2-medium | secondary_fraction | r2 | 0.608 | -917.465 | 918.073 | 0.000 | 0.170 | 0.88 | 300 |
| progen2-medium | residue_ss | macro_f1 | 0.256 | 0.208 | 0.048 | 0.812 | 0.027 | nan | 44626 |
| progen2-medium | decoder_ec | _skipped: decoder cohort only for zymctrl_ | | | | | | | |
