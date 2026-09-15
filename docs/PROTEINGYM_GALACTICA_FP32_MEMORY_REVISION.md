# Galactica FP32 resource revision — consecutive batches

This is an authorized source and resource-validation revision after the first complete 30B FP32 score stopped with CUDA OOM. It is not a pre-data registration and does not revise the earlier BF16 failure or the original single-batch FP32 qualification. The numerical and scientific protocol remains [Galactica FP32 v2](PROTEINGYM_GALACTICA_FP32_V2.md); that document and its earlier artifacts remain unchanged.

## Defect and scope

The native scorer retained the preceding batch's full-vocabulary logits and log-probabilities while evaluating the next forward. A tiny real-torch CPU weak-reference replay confirmed that overlap and showed that the references disappear when the method returns. This is an avoidable cross-batch peak-lifetime defect, not an accumulating leak after return. The failed GPU trace establishes CUDA OOM, but does not prove that retained tensors were its sole cause or that removing them will suffice for every complete workload.

The source change releases batch-local tensors after copying the batch's totals to CPU. It must not change checkpoints, dtype, TF32 policy, numerical operations, residue targets, batch size, scoring order, variants, seed, exclusions, baselines or bootstrap. The original scorer is retained under its existing Git identity and as a hash-verified independent copy; the original protection manifest and all prior outcomes remain evidence for the original implementation, not evidence that the repaired implementation passed.

## Required validation

- A real-torch CPU regression spans multiple full batches and a final partial batch. At every subsequent forward, the preceding large tensors must have no live weak reference. Totals, ordering and residue accounting must agree with an independent reference. The regression must expose the old implementation's overlap.
- The new full v2 probe retains all 21 case observations and all 45 comparisons against the unchanged `1e-4` ceiling, along with interface, refusal, input-binding and cohort checks.
- The longest eligible synthetic resource check uses **one public `log_likelihood` call containing three consecutive full batches**: 48 sequences at configured batch size 16. Three separate calls would let old local references die and would not test the defect. The same AA20-tiled longest eligible shape is used; these are not DMS variants and no synthetic log-likelihood is retained as a fitness score.
- Record the configured batch size, `n_batches=3`, `n_sequences=48`, finite output count, total elapsed time and CUDA synchronized allocated/reserved peaks over the whole call. These dimensions describe the requested workload, not an independently instrumented count of GPU kernels. Missing or failed evidence cannot be treated as a single-batch fallback. Elapsed time covers all three batches, not one batch or the full DMS campaign.
- A new source commit and independent frozen identity must qualify all four Galactica rungs. Admission for this revision requires the consecutive-batch evidence as well as matching successful execution receipts and exact common cohort/input fingerprints. Original single-batch outcomes cannot substitute for it. A complete 30B score still requires its own successful full artifact; partial rounded log correlations are not a cohort result.

Existing snapshot and outcome files are not overwritten. The newly prepared receipt is separate from `fp32_freeze.json`, `freeze.json` and `diagnostic_freeze.json`. Any uncertain freeze retains its attempted identity and no-redispatch guard. RITA remains an independent v1 score and does not inherit this Galactica resource revision.
