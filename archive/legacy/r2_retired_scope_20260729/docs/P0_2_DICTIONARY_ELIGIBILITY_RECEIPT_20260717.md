# P0-2 dictionary eligibility receipt

**Implemented:** 2026-07-17  
**Scientific status:** no real eligibility receipt exists; P0-2 remains open

This is the fail-closed adjudication contract for the exact-cache dictionary
panel frozen in `P0_2_DICTIONARY_PROTOCOL_20260717.md` and
`p0_2_dictionary_controls_production_profile.json`. It does not adjudicate the
preliminary online TopK queues.

## Complete input panel

The entry point is:

```bash
python scripts/66_adjudicate_dictionary_panel.py \
  --spec configs/p0_2_dictionary_gate_spec.executed.json \
  --output-dir results/npj_revision/p0_2/eligibility
```

The executed spec must enumerate exactly:

- three production activation-cache manifests: ProtGPT2, ZymCTRL and
  ProGen2-medium;
- six validation-only screening runs: ReLU/L1 SAE and gated SAE for each
  model, seed 20260717;
- 36 full runs: four methods by three models by seeds 17, 29 and 43; and
- the separately hash-bound mask-validation receipt.

Each screening/full entry binds its run manifest, result and selected
`best.pt` by path and SHA-256. The adjudicator opens and hashes each file. It
also opens each cache manifest, token-selection file and train/validation/test
source manifest; checks their manifest hashes, exact row budgets, split
separation, production status, tokenizer fingerprints, mask contract, layer
geometry and one-shard-per-split/layer inventory; and requires the run
manifest to agree with those identities. Cache activation arrays are not
rehased by this final aggregator: the production runner already performs that
expensive validation before training, while its run manifest binds the cache
manifest/content and every shard digest.

No receipt directory is created if a seed, method, model, checkpoint, source
cohort, cache, screening result or provenance field is missing or invalid.
Preflight, smoke, synthetic, incomplete and test-leaky artifacts are hard
errors. Publication is write-once: an existing output directory is never
overwritten.

## Frozen gates

The code checks the dated protocol text before applying its numbers. For every
model, method and seed it requires:

- passing padding-invariance and all-valid-equivalence tests on the exact
  dictionary module SHA used by the run;
- mean held-out test FVU strictly below 0.50;
- at least 75% of layers with held-out FVU strictly below 0.50;
- median dead-feature fraction strictly below 0.50;
- exactly one test evaluation after validation-only selection, with zero test
  accesses during training;
- the exact 100,000-row held-out valid-token budget and production padding
  contract; and
- for ReLU/L1 and gated methods, the frozen screening selection whose
  validation L0 is in the inclusive interval [115.2, 140.8]. TopK instead
  binds `k=128`; dense binds rank 128. There is no fabricated held-out L0
  threshold.

The intersection of layers with dead fraction strictly below 0.70 in all
three seeds becomes that model/method's downstream layer allowlist. A further
subset may be used only if it is frozen before test-set biological outcomes.
No per-layer FVU threshold beyond the stated 75% rule was added because the
protocol does not specify one.

## Status semantics

Scientific quality failures are valid results, not malformed inputs. A fully
observed panel therefore emits a receipt with root `status: "complete"` even
when one or more dictionaries fail. `p0_2_panel_eligible` is true only if all
12 model/method dictionaries pass; downstream code must not use that aggregate
as its authorization test. It must select the exact
`model_method_adjudications` row and require:

- `status: "atlas_eligible"` and `atlas_eligible: true`;
- all three required seed rows passing;
- exact run-manifest, result and selected-checkpoint SHA-256 values;
- exact split-cohort and cache identities; and
- requested layers contained in `eligible_downstream_layers`.

This distinction resolves an ambiguity in the frozen protocol: alternative
methods are independently eligible and may legitimately fail, so a failed
alternative must not invalidate an otherwise passing TopK dictionary. The
receipt reports both individual and whole-panel status.

The two published files are:

- `p0_2_eligibility_receipt.json`, schema
  `r2_p0_2_eligibility_receipt_v1`; and
- `p0_2_eligibility_receipt.manifest.json`, which records the receipt SHA-256
  and only the immutable input hashes for the complete panel.

`require_eligible_model_method` in `src/revision/dictionary_gate.py` is the
canonical downstream validator. It returns one verified model/method entry
and supports exact all-seed run-manifest/checkpoint hashes, cohort hashes and
requested layers as consumer-side constraints.

## Exact-cache TopK compatibility loader

The eligible TopK artifact is the exact-cache runner's selected `best.pt`, not
an online-queue `clt.pt` and not a checkpoint-manifest digest. The best
checkpoint contains a `WindowedTranscoder` state with:

- `encoder_weight` / `encoder_bias`;
- one `decoder_weight.<layer>` tensor per layer; and
- `decoder_bias`.

These have the same ReLU-then-TopK encoder and windowed decoder equations and
tensor orientations as `CLTForTraining.W_enc`, `b_enc`, `W_dec` and `b_dec`.
`load_eligible_topk_clt` first calls the canonical receipt validator, verifies
the exact chosen `best.pt` hash, candidate ID, profile/cache prefixes, training
step, geometry and finite tensors, then maps those tensors without numerical
conversion or duplication into a frozen `CLTForTraining` instance. Its
returned provenance records the receipt, run manifest, checkpoint, cohort,
geometry, seed and allowlisted layers. A CPU parity test checks identical
feature codes and decoder outputs.

## Mask-validation receipt

The aggregator consumes, but does not fabricate, a receipt with schema
`r2_p0_2_mask_validation_receipt_v1`. It is complete only when it records the
tested module and test-file descriptors, pytest exit code, confirmatory flag
and both exact outcomes:

- `test_valid_token_cache_is_padding_invariant_and_all_valid_equivalent`; and
- `test_windowed_metrics_are_padding_invariant_end_to_end`.

The tested module SHA must equal every screening and full run's module SHA.
A completed failing mask receipt yields negative scientific eligibility; an
incomplete, mismatched or nonconfirmatory receipt is rejected.

Produce this separate receipt with the same Conda interpreter used for local
verification and the module SHA recorded by the screening/full run manifests:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python scripts/76_write_mask_validation_receipt.py \
  --expected-module-sha256 <run-module-sha256> \
  --output results/npj_revision/p0_2/mask_validation_receipt.json
```

The producer invokes exactly the two named pytest nodes in one process, parses
their JUnit outcomes, and hashes both the canonical module and test file before
and after pytest. It publishes no receipt after collection/runtime errors,
unexpected tests, inconsistent exit status, source changes or a module-hash
mismatch, and it refuses to overwrite an existing receipt. A normal assertion
failure is instead preserved as the complete negative scientific outcome
required by the consumer contract.

## Claim boundary

A passing row authorizes only the named dictionary seeds and allowlisted
layers for the next prespecified analysis. It does not establish conserved
biology, a biological-primitive dictionary, single-feature causality, an
attention-sink mechanism or successful steering. Until all real artifacts
exist, the example gate spec remains a non-executable path/hash template and
P0-2 cannot be reported as passed.
