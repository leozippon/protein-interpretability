# Exact-cache r3 execution environment

This directory preserves the environment observed in pod
`damoxing-zhk-zipbio-master-0` while the code-bound P0-2 exact-cache r3 queue
was running on 2026-07-17. `environment_receipt.json` binds the observation to
the archive, code manifest, profile, runner and launcher hashes.

`observed_pip_freeze.txt` is an audit receipt, not a portable lockfile. In
particular, its `vllm` row references a pod-local wheel path that must be
deposited or replaced by a licensed immutable artifact before P0-1 can pass.
The receipt improves provenance for this run but does not reconstruct missing
historical environments or constitute the final DOI-backed release.

- Environment-receipt SHA-256:
  `3dce112700e17ec0bda8534527f731f67bf31b0946f9408d19075759ea6e17dc`
- Observed-pip-freeze SHA-256:
  `02d5b4cfd98079188bfdf7d20382d5e5d62f6d44412e92d59129f7d45eefbafa`
