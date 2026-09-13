# Native ProteinGym extension — pre-data contract

**Status:** protocol-frozen v1 before any new DMS ρ; no new model fitness scores. Machine-checkable freeze receipt: `logs/native_dms_extension/freeze.json` (gitignored runtime record). This document does not embed that receipt's hashes or claim completed ProteinGym scores. **Authority:** `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` remains canonical for existing/recorded DMS findings. This document is only the contract for a *new* Galactica / RITA scoring door. It does not amend EXP-R2-224, EXP-R2-225, or F10, and it does not restate the project summary.

The helper is `scripts/transfer/native_dms_extension.py`. Arithmetic is imported from `src.transfer.scale_comparison`. Native scorers are `src.transfer.galactica_fitness` and `src.transfer.rita_fitness`. Queue draws use `src.transfer.fitness.load_assay`. Stage 20 remains the model-level scorer; this script does not copy that loop.

## Models and roles

| Arm | Role | Dtype | Path |
|---|---|---|---|
| `galactica-1.3b`, `galactica-6.7b`, `galactica-30b` | Dual-mode-qualified checkpoints on this supplement | bfloat16 | `galactica_fitness.CHECKPOINTS` |
| `galactica-125m` | Small-scale dual-mode-unidentified reference, not a fitness verdict | bfloat16 | same table |
| `rita-xl` | Independent single point, only if its interface gate passes | float32 | `STAGED_ARMS['rita-xl'].path` |

Qwen and ProteinGLM are out of this round. RITA is not folded into the Galactica ladder and does not rewrite frozen EXP-R2-225 endpoints. LOOKUP remains an external UniRef50 profile baseline, not a retrieval bound, for both families.

Galactica scores only residue tokens between `[START_AMINO]` and `[END_AMINO]`. RITA uses the tokenizer-native encoding with terminal EOS (id 2); the first token is not a target. RITA Hugging Face length 27 including unsupported Z/26 is not the model output width 26. Scorable ids are AA20 3..22 plus EOS 2.

## What each phase measures

**Probe is not a DMS result.** It is a synthetic interface check, then a freeze of which LOOKUP assays this arm can render without truncation.

Synthetic sequences: Galactica uses variable-length `MKT` and the twenty-letter AA20 string; RITA uses equal-length `MKT` / `MRT`. Checks, in order:

1. Finite totals from the public scorer.
2. Batch totals versus single-sequence totals on the same weights.
3. Two loss checks on the same batch, both under local `torch.inference_mode`. Keep an independent FP32 shifted-CE reference computed from logits. Separately call the model's native `forward(labels=...)` and require a `loss`; `loss` times the true scored-target count must match the reference. Missing loss is failure, not a fallback to the reference. Galactica labels keep `scored_positions` and set the rest to `-100`. RITA labels are the default full encoding including EOS. RITA also compares a forward with no mask against all-ones. Galactica forwards set `use_cache=False`. Author code is not patched. A CPU pass does not certify H200 low-precision native loss; a disagreement is reported as evidence and the threshold is not moved.
4. Direct over-context and illegal amino-acid inputs must raise. Failure is failure: the probe does not write `probe_passed`, does not emit PASS, and does not move a threshold.
5. After `token_lengths` has frozen the eligible assays, one configured-batch forward is required at the longest eligible wild-type residue length. The sequences are an AA20 tiling truncated to that length. They are not truncated DMS variants, carry no labels, and their log-likelihoods are not retained as scores. The public scorer must return a finite total for every row in the batch, and the rendered token length must equal that assay's eligible `max_tokens` and remain inside context. Elapsed time is measured after a CUDA synchronize when the model is on CUDA; peak allocated/reserved bytes are recorded only then. A CPU stub records those memory fields as unmeasured and does not fabricate GPU values. If no assay is eligible, the probe refuses and writes no passing file.

Default `--batch-size` is 16. If and only if this pre-DMS longest-shape forward raises an explicit CUDA OOM, an operator may reduce batch size along 16→8→4→2→1 and re-run the complete probe before any new DMS ρ. Other errors are not treated as OOM. Dtype, queue, scoring direction, and tolerances are not changed. There is no automatic retry or scheduler. Each arm's later stage-20 `batch_size` must equal the batch size of its successful probe. Galactica rungs may differ in batch size for resource reasons; their scientific inputs, support, and dtype stay shared.

Fixed tolerances: FP32 per-target absolute difference ≤ `1e-4`; low-precision batch-versus-single ≤ `0.02` nats/target.

**Analyse consumes later stage-20 score files.** It reports raw Spearman, MODEL−LOOKUP, and MODEL−BLOSUM62, plus LOOKUP and BLOSUM62 raw means on the same support and the same family weights. BLOSUM62 is not a capability gate. The script does not write `acquired` or `retrieval_bounded`, does not rank across strata or families, and does not treat a rung difference as a parameter-count effect.

## Frozen queue, support, and fingerprints

LOOKUP order is the original `assays` list. Each assay is redrawn with `load_assay(..., n=1000, seed=20260807 + original_index)`. The index is not reset after a skip. `mutant_digest`, `n_variants`, and `wildtype_id` must match LOOKUP. Probe requires `--wildtypes` and `--wildtypes-fasta` with no guessed path. The JSON catalogue stores `qNNNNN` identifiers, lengths, and `assay_to_wildtype`; it does not store sequences. Recovery is still `wildtype_of` on the same CSV `load_assay` reads. The original `wildtypes.faa` supplies the full wild-type string for that id; missing file, missing id, duplicate id, or a same-length string mismatch is failure. Length and cluster checks remain. `mutant_digest` fingerprints mutation strings only and does not bind full sequences or DMS labels. Historical CSV hashes do not prove old DMS labels. Stage 20 does not consume the FASTA.

After the scorer loads, `token_lengths` is recorded on every drawn variant. An assay is excluded only when the maximum rendered length exceeds that checkpoint's context. Truncation is refused. The longest remaining eligible wild type then supplies the residue count for the synthetic batch-shape gate above. Galactica's four rungs are one common-support group: disagreeing exclusions are a hard error, not an intersection. RITA is a separate group with its own support.

Fingerprints: SHA-256 of LOOKUP, of `wildtypes.json`, of `wildtypes.faa`, and of each assay CSV. Analyse refuses a LOOKUP file whose hash disagrees with every probe, even if assay identifiers and digests are unchanged. Galactica probes must share the FASTA fingerprint and are compared per assay on CSV hash, wild-type id and sequence, variant count, digest, cluster, and sampling seed/cap, not only on support. Galactica / RITA stage-20 score files carry per-row `csv_sha256` from the same ProteinGym directory `load_assay` used, plus `input_fingerprints.wildtypes_sha256` of the cell `wildtypes.json`. They do not grow a FASTA field. Analyse requires the JSON/CSV hashes to match the probe. Default arms are unchanged. Probe artefacts also record loader facts, imported Python / PyTorch / Transformers versions, dtype, stratum, batch size, and the code paths actually called.

New fitness numbers must not rewrite this queue or these tolerances.

## Statistics

Family means: per-assay Spearman, then the mean inside a wild-type family, then an equal-weight family bootstrap, 2000 resamples, 95% percentile, through `unit_mean_record` / `endpoint`. Paired Δρ uses the same original units and is not the difference of two marginal confidence intervals.

Seed base `20260913`. Channels: raw `+0`, MODEL−LOOKUP `+100`, MODEL−BLOSUM62 `+200`, LOOKUP raw mean `+300`, BLOSUM62 raw mean `+400`. The RITA group adds `+1000`. Inside `endpoint`, rung `i` draws at `seed+i` and pair `j` at `seed+10+j`.

Galactica pairs: `125m → 1.3b` is reference only; the qualified ladder is `1.3b → 6.7b → 30b`. RITA has no adjacent pair.

`align_dms` is called with a `DmsCensus` built from the frozen probe queue. The exclusion reason is the stage-20 over-context sentence (matched on `exceeds this arm's context`). NaN is refused by `write_json`.

## Failure rules

- Missing any *requested* probe or score path is an error. A three-rung Galactica subset is not summarised.
- Probe failure writes no passing `interface_check.json`.
- A probe with no eligible assay after context exclusion is a failure.
- Mixed dtype or mixed scoring stratum inside a group is refused. Score `batch_size` must match that arm's probe; Galactica rungs are not required to share one batch size.
- Digest, wild type, variant count, CSV hash, wildtypes hash, sampling seed/cap, batch size, or skip-set disagreement between probe and score is refused. Missing `wildtype_id` or `n_variants` is refused; two missing values that happen to compare equal are not a pass.
- nnsight / wandb are not imported and are not auto-installed. A failed `torch` or `transformers` import stops and names the missing dependency.

## Runtime stacks

These are different environments. Record what the process actually imported.

| Host | Python | PyTorch | Transformers |
|---|---|---|---|
| Compute `ct` | 3.11.14 | 2.9.1+cu128 | 4.57.3 |
| H200 `/opt/ac2` | 3.12.7 | 2.7.1+cu128 | 4.52.4 |

A CPU pytest pass is not an H200 pass. Remote work uses uploaded script files, not multi-line `python -c`.

## Run order

1. CPU acceptance of this helper and its tests.
2. Freeze this protocol and the helper code (protocol-frozen v1 before any new DMS ρ; verify `logs/native_dms_extension/freeze.json`).
3. H200 `probe`, one arm at a time, writing `interface_check.json` and freezing the actual support manifest **before any new DMS ρ**. If the longest-shape batch OOMs, reduce batch size along the ladder and re-run the complete probe; do not start scoring until that probe has passed.
4. Stage 20 full model-level parallel scoring of that frozen queue.
5. This script's `analyse`.
6. Acceptance, archive, release of any temporary H200 allocation, retain one GPU.

RITA's condition for inclusion is the interface gate, not whether its later Spearman looks good.

## Limitations

Character/residue rendering, tokenizer, and context still differ across families. Common support is per group, not a cross-family ranking set. LOOKUP is not identified as either model's pretraining corpus. Dual-mode qualification is a separate existing record. This protocol-frozen v1 does not allocate GPUs and does not claim completed ProteinGym scores. A campaign identifier, if any, lives only in the freeze receipt, not in this document.
