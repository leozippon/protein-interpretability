# Text amino-acid string-control ProteinGym protocol v1

**Status:** census, probe, score, and analyse are implemented. File presence, tokenizer loading, a tiny census, this document, and a launched queue are not ProteinGym capability results. They do not admit an experiment and they do not carry numbers until the queue finishes.

This is an independent controller for thirteen text checkpoints scored as raw uppercase AA20 strings. It is not native protein semantics, not a panel door, and not a widening of the frozen three-arm ProteinGym run. The research role is `text-aa-string-control`. ByGPT5 is the residue-aligned subset. GPT-2, Qwen, Llama, and DialoGPT are string-level BPE controls.

## Encoding rule amendment (user-ordered, 2026-09-20)

The protocol id remains `text-aa-fp32-v1`. The encoding rule changed by user order, not as a silent fallback.

The estimand is the sum of the conditional log-likelihood of every target token of a raw uppercase AA20 string after the declared document-boundary token (or with no prefix for ByGPT5). Mutant minus wild type. No length normalisation. No biological prompt. No trailing EOS scored. Float32, batch size 1, TF32 off.

Merged BPE is allowed. `encode_text_aa` no longer raises `TextAAEncodingError` solely because `len(target_ids) != len(sequence)`. Protein BPE (ProtGPT2) is already scored as a summed token likelihood; text checkpoints use the same token-sum. That sum is not the per-residue functional protein and joint models report under `joint_modes.verify_one_token_per_residue`.

Still refused: empty strings, characters outside AA20, UNK/special/forbidden ids, a failed strict decode round-trip, and a conditioning id inside the targets.

Every scored sequence and assay records:

- `one_token_per_residue` when `len(target_ids) == len(sequence)`
- `merged_bpe` otherwise
- `n_residues`, `n_target_tokens`, and `residues_per_token`

`n_prefix_tokens` is 0 or 1 from the declared prefix. It is not `len(ids) - len(sequence)`, which is wrong under merges. `n_scored_tokens` remains `len(ids) - 1` (the first token is context).

Analyse does not emit `acquired` or `retrieval_bounded` verdicts that would treat merged-BPE text scores as the same functional as residue protein models.

## What is implemented now

Census, probe, score, and analyse.

For each named checkpoint the census loads tokenizer and config, never weights, and encodes the frozen ProteinGym request: the wild type plus every variant already selected under seed `20260807 + original LOOKUP index` and cap 1000. Variants are not redrawn. An assay that fails a declared encoding restriction (`TextAAEncodingError` for illegal alphabet and the remaining refusals above), or that exceeds that checkpoint's hard context, is excluded as a whole. Merged encodings are admitted. Tokenizer internals, schema, configuration, memory, and unknown backend errors fail the census immediately; they are not assay exclusions. The original assay list, mutation digest, CSV identity, and failed sequence roles remain in the ledger. Model and assay `n_encode_ok` count sequences that encoded, including those that later exceed hard context. Over-window rows are successful encodings that are not admitted to the window, not a second encoding failure. `n_encode_fail` is the declared-restriction count; `n_exceeds_hard_context` is the window count; `n_failed` is their sum; `n_eligible_sequences` is the window-admitted remainder.

GPT-2, Qwen, and Llama use the config hard context as the application window. ByGPT5 has no absolute context; `documented_context is None` means no hard context, not an unspecified value, and a later `n_positions` must fail rather than silently become a cap. Its window is the maximum legal prefix-plus-target length on the frozen request. Generation defaults such as 20 tokens, and older budget numbers such as 384, are not windows. If no strictly legal sequence exists, census fails rather than inventing a window.

The longest resource-probe identity is the longest legal actual input that fits the final window, with ties broken by LOOKUP index, then wild type before mutant, then the original mutant list. Only identity hashes and lengths are stored. Sequences that miss the hard context cannot be probe inputs.

Support sets are declared before any score. Each model has a native admitted list. The thirteen-way intersection exists only when all thirteen models were censused. Separate family intersections exist only for GPT-2 four rungs, Qwen2.5 three rungs, and ByGPT5 three rungs. DialoGPT-small is not a fifth GPT-2 scale rung. Qwen3-8B-Base is the Base checkpoint, not Instruct, and is not mixed into the Qwen2.5 scale axis. Llama, Qwen3, and DialoGPT keep native and all-thirteen readings. Scores do not choose support sets.

Scoring is float32 from the checkpoint, batch size 1, TF32 off. The numeric reference is `1e-4` nats per target against independent shifted cross-entropy.

`score` runs qualification in-process and writes a probe artefact, then scores. A failed probe refuses to emit a scientific score.

Qualification uses five labelled cases: `A`; the shared AA20 standard string; the wild type of the first admitted assay in original order; that assay's first selected mutant; and the census longest legal actual input. Each case accounts total log-likelihood against independent shifted cross-entropy and the target count.

The resource gate is one public `log_likelihood` call that repeats the longest actual input three times, recording the whole-call synchronous CUDA allocated and reserved peaks, finite counts, and wall time. Three independent calls are not a substitute. Synthetic tiled sequences are not a substitute.

A non-finite model score or a missing score is an instrument failure. It leaves a diagnostic null on the original fixed ledger and blocks scientific admission for that model. It is not an ordinary constant prediction. A finite constant prediction or finite constant measurement is a real Spearman of null. If a later fixed support set still contains undefined rows, the complete fixed-set point and interval are null. Defined-only family means and pairwise deltas that require both sides defined are reported with their own denominators and are not the fixed set.

One model's qualification failure does not stop another already-qualified model from native scoring. It does stop any claim that all thirteen models are complete.

Analyse is CPU. It does not call the legacy ProteinGym Spearman helper and does not invent FDR. Family means and paired deltas reuse the existing unit-mean and paired-delta helpers only on matching finite subsets and the correct units.

Bootstrap seed base is `20260913`, text offset `2000`, 2000 resamples. Stable blocks are: native models in the declared thirteen-order as indices 0–12, all-thirteen common as 13, GPT-2 family as 14, Qwen2.5 family as 15, ByGPT5 family as 16. The draw seed is `base + 2000 + 10000 * support_index + 100 * metric_index + contrast_index`. Metrics are raw 0, model minus LOOKUP 1, model minus BLOSUM62 2, and within-family adjacent pair delta 3. Per-model contrast indices follow the same thirteen-order. Family pair contrasts use the predeclared adjacent pairs of that family only. Null or failed rows do not reorder seeds. The endpoint helper's internal `+10` pair offset is not reused, because it would collide across thirteen rungs.

## How to invoke

The dedicated command accepts the campaign runner's injected `--device` and `--out`.

Census is tokenizer-only on CPU:

```
text_aa_dms.py census --lookup <lookup.json> --wildtypes <wildtypes.json> --wildtypes-fasta <wildtypes.faa> --proteingym-dir <DMS_ProteinGym_substitutions>
```

Score one checkpoint after a census. It writes the probe artefact first, then the score, and refuses the score if the probe fails:

```
text_aa_dms.py score --model gpt2 --census <text_aa_census.json> --lookup <lookup.json> --wildtypes <wildtypes.json> --wildtypes-fasta <wildtypes.faa> --proteingym-dir <DMS_ProteinGym_substitutions>
```

Analyse is CPU and refuses a missing score:

```
text_aa_dms.py analyse --census <text_aa_census.json> --lookup <lookup.json> --score gpt2=<text_aa_score.json> ...
```

The campaign table is `scripts/transfer/campaign_text_aa_string_control.tsv`: one CPU census of all thirteen, then thirteen score cells on two GPUs (smaller first, `qwen2.5-32b` last), then one CPU analyse of all thirteen.

## Freeze and hosts

Full tokenizer census and scoring run on H200. The local workstation is only for small synthetic requests and tiny interface tests.

Freeze census, probe, score, and analyse code, the boundary table, and the queue configuration with the existing `run_transfer_h200.sh --pin <commit> --freeze-only` path. Write artefacts under ordinary results. Do not write products back into an already frozen snapshot.

After records-only verification, a later manifest snapshot may cite the census path, file digest, support-set digest, and application window. Adding this score path requires a new freeze of that scientific source. Shared encoder, config, and tokenizer fingerprints must match.

Runtime logs and long likelihood vectors do not belong in git. The boundary table, this protocol, and small manifest schemas may be tracked.

Old panel defaults, the three-arm ProteinGym corpus, scoreable native arms, and the three budget gates stay unchanged.
