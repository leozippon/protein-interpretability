# ProGen3 Stage-46 homologous-context remeasurement

Both withdrawn ProGen3 verdicts are confirmed on corrected scores, and so are both withdrawn fifteen-checkpoint correlations. One sub-quantity inside the confirmed ProGen3-3B verdict reverses: in the low-identity, low-overlap stratum the paired-concordance 95% lower bound falls from +0.52239 to +0.43284 and no longer sits above one half. The compound gate outcome is unchanged because that clause already failed on its fractional-reduction term.

The [canonical audit](INTERPRETABILITY_TRANSFER_AUDIT.md#0-objective-and-current-programme) carries the retraction this document answers. Historical receipts are preserved, not overwritten.

## The defect

`_forward_progen3` in `scripts/transfer/46_context_homologue.py` assigned each batch row's index to `sequence_ids` (line 677 of the withdrawn code). On this model `sequence_ids` is a learned-embedding index: `ProGen3Model.forward` computes `inputs_embeds = self.embed_tokens(input_ids) + self.embed_seq_id(sequence_ids)` over a `nn.Embedding(config.max_num_sequences, config.hidden_size)` with `max_num_sequences` 512, so row *i* received `embed_seq_id.weight[i]` added to every one of its token embeddings. The native `ProGen3BatchPreparer` assigns `torch.zeros(len(tokens))` in `prepare_clm` for an independent sequence and pads `sequence_ids` with 0, so every batch position above the first was scored on an input the native interface does not produce. No index exceeded the embedding's range, which is why nothing raised.

Only `progen3-*` arms reach that function: `packing_of` in `src/transfer/context_homologue.py` maps them to `PACKING_PROGEN3` and sends every other arm through the padded, attention-masked path or its own packing. Stage 46's only numerical guard was a singleton self-check, and a batch-position defect cannot reach a batch of one — the two self-check NLLs reproduce bit-identically across the withdrawn and corrected runs (ProGen3-112M 3.4216549396514893, ProGen3-3B 0.44272053241729736 nats per scored token).

A second, separate reason a batched row is not a native score survives that repair. `SparseMoeBlock.forward` flattens batch and position into one token axis and runs one gather-and-GEMM per expert over the whole flattened batch, so the accumulation producing a token's own output depends on how many other tokens routed to its expert, and this release has no kernel above float16. Measured against singleton scoring of the same 16 rows of 476 to 1,022 tokens in bfloat16 on an H200, with the corrected zero assignment in place, the maximum absolute difference is 0.02464 nats per scored token for ProGen3-3B at batch size two (mean 0.00658) and 0.02796 for ProGen3-112M at batch size eight (mean 0.01252). Two further ProGen3-112M probes on a local L20 locate the dependence. Eight identical copies of one 770-token row in a single batch return exactly the same value as each other, while that batch's value sits 0.02444 nats per token from the row scored alone and a batch of two sits 0.00267 away: the dependence is on batch extent, not on batch position. Sixteen equal-length rows, which need no padding at all, still disagree with their singleton values by up to 0.07179 nats per token, so padding is not the cause either. This resolves what the retraction recorded as an unresolved residual.

### What it invalidates

The ten score files of frozen run `20260918221340_b43d0e29e7de`, and every aggregate derived from them. Under the manifest's batch sizes, and with plan order fixing each unit's batch position, the rows that carried a nonzero sequence-ID bias are:

| Arm | Batch size | Scored rows affected | Pooled units affected | Decisive-stratum units affected |
| --- | ---: | ---: | ---: | ---: |
| ProGen3-112M | 8 | 875 / 1,000 | 700 / 800 | 58 / 67 |
| ProGen3-3B | 2 | 500 / 1,000 | 400 / 800 | 34 / 67 |

The remaining rows are not thereby native values. Comparing the withdrawn against the corrected per-row NLL, rows that sat at batch position 0 still differ by up to 0.06238 (ProGen3-112M, position-only) and 0.19789 (ProGen3-3B, monomer-shuffled) nats per scored token, because a perturbed neighbour changes the expert reduction for the whole batch. Across all five conditions the withdrawn-minus-corrected difference reaches 0.28231 nats per token for ProGen3-112M and 0.34304 for ProGen3-3B; its mean absolute value per condition is 0.00314 to 0.02454 and 0.00248 to 0.03256 respectively. On ProGen3-3B the difference carries a common-mode sign, mean +0.01956 (position-only), +0.01951 (unrelated) and +0.01489 (homologue) nats per token, which is why the paired contrasts moved much less than the individual scores.

### What it leaves intact

The fourteen other Stage-46 expansion arms never call this function. Both singleton self-checks are unaffected and reproduce exactly. The 2026-09-23 [context-rescue](D1_CONTEXT_RESCUE_RESULTS.md) ProGen3 scores record batch size one and are unaffected by the assignment, as are the profile-increment analyses built on them. Native ProGen3 scoring and generation paths that use the native preparer never had the defect, so the ProteinGym, MegaScale and context-information results stand. The scope matches the retraction's; this remeasurement neither widens nor narrows it.

## The repair

One change, in the score path that owns the endpoint. `_forward_progen3` now builds the native independent-sequence ids — zero for content and padding, at every batch position and every batch size — and `run_score` refuses a batch size above one on this packing, before any weight is loaded, naming the measured offset and the flag. The batch size is recorded in every score artefact rather than left to be recovered from a queue manifest, which is how the withdrawn run's batch sizes had to be established.

`tests/test_context_homologue_batch_invariance.py` holds the conditions that were missing. The native-id condition is checked for single, multi-row, mixed-length and reordered batches and is cross-checked against the vendored preparer's own value rather than a literal; a row's model input is required to be a function of that row alone; the score path's refusal is exercised as a negative path, together with the batch size one that passes it; and the batched-versus-singleton target-NLL equality is asserted on the padded path that Stage 46 does still batch, with a deliberately batch-dependent stand-in model confirming the comparison has teeth. On real weights and a GPU, a ProGen3 singleton score is required to be independent of what was scored before it. Run against the withdrawn code, the native-id condition fails with row maxima 0, 1, 2 — these tests would have caught the defect.

## The remeasurement

Frozen run `20260923183217_734c7814635d`, twelve cells over six idle H200 cards in three allocations, all `exited-ok` with `# FAILURES 0` and `# NO-RECORD 0` in each of the three lane manifests. Nothing in the design moved: the same cohort digest `33707cee59c523dec945f9cc6f815bf5e5a1610eeff6e40ca2f16ce3134a7532`, the same frozen plans (digests `92cf3a0bb9133bfbb1e8588688e7d3e06e4d0a56b2e319f7f5630b20c213048e` and `ba7edc6008e942c2ee3ed9a3b70e6500cdca82e21a5ce942b7f484d0ca83d8e7`, 1,000 units each), the same five conditions in the registration's order, the same bfloat16 kernel, and the same analysis weighting — 2,000 bootstrap resamples at seed 20260826, resampled over the target's near-duplicate group, with units averaged equally inside each stratum. The only changed setting is `--batch-size 1` in place of eight and two. All ten score files carry 1,000 units and record that batch size.

The field named AUROC is a within-target paired win probability with half credit for ties, reported here as paired concordance. The fractional reduction is (unrelated − homologue) / position-only, in the same nats-per-scored-token unit as its denominator. Pooled estimates use the 800 retained units in 710 near-duplicate groups; the decisive stratum is the bottom local-overlap tercile of the below-30%-identity band, 67 units in 67 groups.

### ProGen3-112M — `no_gain_at_this_budget` confirmed

| Endpoint | Stratum | Withdrawn | Remeasured |
| --- | --- | ---: | ---: |
| Paired concordance | pooled | 0.48500 [0.44888, 0.51991] | 0.48000 [0.44471, 0.51585] |
| Fractional reduction | pooled | −0.00097 [−0.00290, +0.00099] | −0.00134 [−0.00329, +0.00059] |
| Homologue − unrelated | pooled | +0.00311 [−0.00308, +0.00917] | +0.00436 [−0.00171, +0.01046] |
| Paired concordance | decisive | 0.47761 [0.35821, 0.59701] | 0.43284 [0.31343, 0.55224] |
| Fractional reduction | decisive | +0.00159 [−0.00448, +0.00806] | +0.00140 [−0.00502, +0.00819] |
| Homologue − unrelated | decisive | −0.00488 [−0.02517, +0.01424] | −0.00415 [−0.02544, +0.01622] |

All three gate clauses fail exactly as before: the pooled concordance lower bound stays below one half, the pooled fractional-reduction lower bound stays below zero, and the decisive stratum clears neither. The verdict is confirmed. Its licence is unchanged: a bounded negative about this frozen decoder at the measured context counts, not a claim about in-context homologue conditioning in general.

### ProGen3-3B — `in_context_copying_and_local_overlap` confirmed, with one sub-quantity reversed

| Endpoint | Stratum | Withdrawn | Remeasured |
| --- | --- | ---: | ---: |
| Paired concordance | pooled | 0.80000 [0.76962, 0.82875] | 0.79875 [0.76875, 0.82825] |
| Fractional reduction | pooled | +0.17830 [+0.16018, +0.19606] | +0.17825 [+0.16007, +0.19588] |
| Homologue − unrelated | pooled | −0.53093 [−0.58772, −0.47430] | −0.52851 [−0.58528, −0.47174] |
| Monomer-shuffled − homologue | pooled | +0.60459 [+0.54661, +0.66175] | +0.60413 [+0.54637, +0.65995] |
| Paired concordance | decisive | 0.64179 [0.52239, 0.74627] | 0.55224 [0.43284, 0.67164] |
| Fractional reduction | decisive | +0.00816 [−0.00631, +0.02244] | +0.00696 [−0.00877, +0.02205] |
| Homologue − unrelated | decisive | −0.02809 [−0.06615, +0.01050] | −0.02541 [−0.06629, +0.01698] |

The two pooled clauses hold and the decisive-stratum clause fails, as in the withdrawn verdict, so the compound outcome and its licence are confirmed: the gain is in-context copying and local overlap.

Inside that confirmed verdict one quantity reverses its resolution. The withdrawn decisive-stratum concordance had a lower bound of +0.52239, above one half; the remeasured bound is +0.43284, spanning one half. The clause failed either way on its fractional-reduction term, so the gate is not affected, but any statement that ProGen3-3B's paired concordance is resolved above one half in the low-identity, low-overlap stratum is withdrawn by this remeasurement and is not reinstated. The decisive stratum is 67 units in 67 groups; it is the smallest support in this design and the one where the correction moved most.

### The fifteen-checkpoint correlation — both coefficients confirmed

The withdrawn coefficients came from `manuscript_cross_result_analysis.py`, which correlates each checkpoint's Stage-46 paired concordance against its unconditional any-Pfam recognition rate over the fifteen checkpoints that have both. Only the two ProGen3 rows were affected, so `scripts/transfer/analyse_s46_progen3_remeasurement.py` replaces exactly those two and recomputes the same statistic; it first reproduces the withdrawn coefficients from the historical inputs and refuses to report anything if that reproduction misses them. It reproduces −0.610714 and −0.551175 and the recorded eight lineage-exclusion ranges (−0.7319 to −0.3091, −0.6991 to −0.3549) exactly.

| Axis | Withdrawn | Remeasured | Leave-one-lineage-out, remeasured |
| --- | ---: | ---: | --- |
| Pooled concordance | −0.6107 | −0.6107 | eight exclusions, −0.7319 to −0.3091, all negative |
| Low-overlap concordance | −0.5512 | −0.5694 | eight exclusions, −0.7167 to −0.2746, all negative |

The resampling unit here is the checkpoint, and, as in the withdrawn analysis, no interval across a checkpoint population is estimated: fifteen is the whole fixed panel, not a sample from one. The pooled coefficient does not move at all because Spearman reads ranks and neither ProGen3 checkpoint changed rank on either axis. Both coefficients are confirmed, and the negative sign is not thereby a law: it is a descriptive association over one fixed panel, and recognition rate is a profile-hit rate over 800 retained attempts, not a measure of function.

## Evidence

Retained under `results/transfer/s46rm_progen3_remeasurement/`:

| Artefact | SHA-256 |
| --- | --- |
| `context_homologue.json` (corrected two-arm analyse) | `cd90106ecc61f8e4e4b3e977882c5a07ab5bc6748a6556ee3cb3a4cb8e281008` |
| `s46_progen3_remeasurement_association.json` | `39dc7930d08c2e144732080bbde282864fb7e5c57d3cfeb33972d645c5388357` |
| `s46rm_withdrawn_comparison.json` | `30ea0bdaa1a62b4cb14f46b1dadbf80d4422b745932477d5b6fcd14f74e60c53` |

The ten corrected score files stay in the frozen run directory. Their digests are, for ProGen3-112M, `90d97d2e6a59b22a5fabde572234a795a00624547b1a376ff050a6195f6fda6a` (position-only), `42b993ecce27c1f3b9e1d7f3a61c7485362076c809913aedaf47329a42dec8c0` (monomer-shuffled), `a8325849116af3aa0e7beb157fc8af57771c7d38bd4c7e88db5b402d92a7e527` (no context), `027ac3194dea04388696e4719065709daad35913641c6a5bfd0c0eeb4c1958e2` (unrelated) and `d6dcc7326e5423aa0d67e636e2d3f14cd88e5671f64e910238d91fb864708e17` (homologue); and for ProGen3-3B, `527b27776d84a62034fba0d9a4f3b4cb8e1b30b71cfbdcb3e8d78b31738a61fd`, `1178e89cf8f802c71ec309ddc9a3c9395c3d9936ceb6ff50f48863e493e4627e`, `805fecdc97befc4c73f4b6f354d74641950cc4786b6513691b29db2f9aaacf62`, `e9ff669b1a2615aa233af87a89ce24db634aaeebe8df8da178d76177b70f4c6a` and `42d68ebcba8c1d973f3a18ab721e9e920fa1663bd98b5a9b395cba8cb82a927f` in the same order. The withdrawn aggregate `results/transfer/s46x_context_homologue_expansion/context_homologue.json`, SHA-256 `e03b5fa50ab4f6ca6b276a3c0bea38172116b05fab0f5bb622fa403f708e0bbf`, remains a historical artefact and supplies the thirteen unaffected checkpoints' retained concordance means.

The three lane manifests are `scripts/transfer/campaign_s46_progen3_remeasure_lane_{a,b,c}.tsv`. Cell durations sum to about 21 GPU-minutes across the three allocations, at 30-second status granularity, with a wall clock of roughly 5.5 minutes. The two batch-dependence probes ran sequentially on two cards of one allocation and were not separately timed; each loaded one checkpoint and scored 16 rows singly and in batches. The batch-dependence probe scripts stay under ignored `logs/d1_s46_remeasure/`.

## Limitations

The withdrawn per-row values cannot be decomposed after the fact into a sequence-ID component and a batch-extent component; the withdrawn-minus-corrected difference reported above is their sum. Correcting the assignment does not make batched ProGen3 scoring valid, and the score path no longer offers it.

ProGen3 has no kernel above float16 on this release, so no full-precision reference exists for either the batched or the singleton value. A singleton score is exactly reproducible on repeat and is the native interface's own quantity, which is what these verdicts rest on; it is not thereby certified against a full-precision computation, and that remains a ProGen3 limitation rather than something this remeasurement resolves.

The correlation replaces two of fifteen rows; the other thirteen carry their retained historical concordance means, which is what a replacement of a withdrawn coefficient requires and not an independent re-derivation of the whole panel. Only the two ProGen3 arms were rescored, so the other fourteen Stage-46 expansion arms' verdicts are historical values here, unaffected but also not re-verified.

Everything a confirmed verdict licensed before, it licenses now and no more. The paired concordance is a within-target win probability under a declared rendering, the fractional reduction is measured against a fixed-filler denominator, and neither identifies a mechanism. The decisive stratum's 67 units in 67 groups are the narrowest support here and carry the widest intervals.
