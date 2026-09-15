# Text amino-acid string-control ProteinGym protocol v1

**Status:** census is implemented. Probe, score, and analyse are specified here and not implemented. File presence, tokenizer loading, a tiny census, and this document are not ProteinGym capability results. They do not admit an experiment.

This is an independent controller for thirteen text checkpoints scored as raw uppercase AA20 strings. It is not native protein semantics, not a panel door, and not a widening of the frozen three-arm ProteinGym run.

## What is implemented now

Census only. For each named checkpoint it loads tokenizer and config, never weights, and encodes the frozen ProteinGym request: the wild type plus every variant already selected under seed `20260807 + original LOOKUP index` and cap 1000. Variants are not redrawn. An assay that fails a declared encoding restriction, or that exceeds that checkpoint's hard context, is excluded as a whole. The original assay list, mutation digest, CSV identity, and failed sequence roles remain in the ledger.

GPT-2, Qwen, and Llama use the config hard context as the application window. ByGPT5 has no absolute context; its window is the maximum legal prefix-plus-target length on the frozen request. Generation defaults such as 20 tokens, and older budget numbers such as 384, are not windows. If no strictly legal sequence exists, census fails rather than inventing a window.

The longest resource-probe identity is the longest legal actual input that fits the final window, with ties broken by LOOKUP index, then wild type before mutant, then the original mutant list. Only identity hashes and lengths are stored. Sequences that miss the hard context cannot be probe inputs.

Support sets are declared before any score. Each model has a native admitted list. The thirteen-way intersection exists only when all thirteen models were censused. Separate family intersections exist only for GPT-2 four rungs, Qwen2.5 three rungs, and ByGPT5 three rungs. DialoGPT-small is not a fifth GPT-2 scale rung. Qwen3-8B-Base is the Base checkpoint, not Instruct, and is not mixed into the Qwen2.5 scale axis. Llama, Qwen3, and DialoGPT keep native and all-thirteen readings. Scores do not choose support sets.

## What later phases must do

Those phases are not present in this commit. Empty success entries are refused.

Scoring is float32 from the checkpoint, batch size 1, TF32 off. The estimand is the sum of every target-token conditional log-likelihood after one declared document-boundary token; mutant minus wild type; no length normalisation. The numeric reference is `1e-4` nats per target against independent shifted cross-entropy.

Qualification uses five labelled cases: `A`; a shared AA20 standard string; the wild type of the first admitted assay in original order; that assay's first selected mutant; and the census longest legal actual input. Each case accounts total log-likelihood against independent shifted cross-entropy and the target count.

The resource gate is one public `log_likelihood` call that repeats the longest actual input three times, recording the whole-call synchronous CUDA allocated and reserved peaks, finite counts, and wall time. Three independent calls are not a substitute. Synthetic tiled sequences are not a substitute.

A non-finite model score or a missing score is an instrument failure. It leaves a diagnostic null on the original fixed ledger and blocks scientific admission for that model. It is not an ordinary constant prediction. A finite constant prediction or finite constant measurement is a real Spearman of null. If a later fixed support set still contains undefined rows, the complete fixed-set point and interval are null. Defined-only family means and pairwise deltas that require both sides defined are reported with their own denominators and are not the fixed set.

One model's qualification failure does not stop another already-qualified model from native scoring. It does stop any claim that all thirteen models are complete.

Analyse does not call the legacy ProteinGym Spearman helper and does not invent FDR. Family means and paired deltas reuse the existing unit-mean and paired-delta helpers only on matching finite subsets and the correct units.

Bootstrap seed base is `20260913`, text offset `2000`, 2000 resamples. Stable blocks are: native models in the declared thirteen-order as indices 0–12, all-thirteen common as 13, GPT-2 family as 14, Qwen2.5 family as 15, ByGPT5 family as 16. The draw seed is `base + 2000 + 10000 * support_index + 100 * metric_index + contrast_index`. Metrics are raw 0, model minus LOOKUP 1, model minus BLOSUM62 2, and within-family adjacent pair delta 3. Per-model contrast indices follow the same thirteen-order. Family pair contrasts use the predeclared adjacent pairs of that family only. Null or failed rows do not reorder seeds. The endpoint helper's internal `+10` pair offset is not reused, because it would collide across thirteen rungs.

## Freeze and hosts

Full tokenizer census runs on H200. The local workstation is only for small synthetic requests and tiny interface tests.

Freeze census code, the boundary table, and the initial queue configuration with the existing `run_transfer_h200.sh --pin <commit> --freeze-only` path. Write census artefacts under ordinary results. Do not write products back into an already frozen snapshot.

After records-only verification, a later manifest snapshot may cite the census path, file digest, support-set digest, and application window. Adding probe or score code requires a new freeze of that scientific source. The census keeps its own source identity. Shared encoder, config, and tokenizer fingerprints must match. New code must not claim the old census freeze as its qualification.

After qualification, a configuration-only snapshot may cite the qualified scientific snapshot; queue cells still execute that scientific snapshot. There may be several manifest snapshots. Two directories are not a limit.

Runtime logs and long likelihood vectors do not belong in git. The boundary table, this protocol, and small manifest schemas may be tracked.

## Controller

The dedicated census command accepts the campaign runner's injected `--device` and `--out`. Census still runs tokenizer-only on CPU, does not call CUDA, and does not treat GPU occupancy as missing scientific data. Default models are the thirteen named above. A subset helper is allowed for tests and must not report a complete thirteen-way support set.

Old panel defaults, the three-arm ProteinGym corpus, scoreable native arms, and the three budget gates stay unchanged.
