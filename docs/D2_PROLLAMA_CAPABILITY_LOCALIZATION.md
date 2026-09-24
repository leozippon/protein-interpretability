# Localizing the ProLLaMA mutation-ranking gain by parameter transplant

The [readout panel](D1_READOUT_RESULTS.md) names the cleanest matched capability difference in the whole 34-arm panel. On one identical 211-assay native support the Llama-2-7B parent's native likelihood ranks mutations at −0.00740 [−0.02713, +0.01226] in within-assay Spearman, ProLLaMA Stage 1's at +0.14604 [+0.11814, +0.17435] and Stage 2's at +0.15865 [+0.13170, +0.18655]. Architecture, tokenizer, rendering, scale and cohort are held fixed across the three; only the training differs, and the parent-to-Stage 1 step carries +0.15344 of the +0.16605 total.

This experiment asks which part of that checkpoint difference is enough to carry the gain. It copies declared groups of Stage 1's parameters into the parent, re-measures the same quantity on the same support, and reports what fraction of the parent-to-Stage 1 gain each group recovers. The [component diagnostic](D1_COMPONENT_DIAGNOSTIC_RESULTS.md) did the analogous thing for the Stage 1/Stage 2 step and found the two stages' embedding and head byte-identical, which collapsed that intervention to a single body contrast. The parent/Stage 1 pair has no such collapse: Stage 1 retrained the embedding and the head alongside every projection.

The endpoint is the model's own likelihood, not a fitted readout. It needs no folds, no penalty grid, no projection and no split seed, so nothing here is conditional on the readout class the [class sweep](D1_READOUT_CLASS_SWEEP.md) and the [depth sweep](D1_READOUT_DEPTH_SWEEP.md) are testing.

## What was declared, and when

Everything in this section was fixed in frozen snapshot `20260924012158_ad680f27d6a8` before any cell ran, and the census below completed before the cell list was written.

**The endpoint.** Per assay, the Spearman correlation between the native mutant-minus-wild-type summed target log likelihood and the measured effect; averaged within wild-type family at 50% identity and then equally over families; with a pointwise 95% percentile interval from 2,000 bootstrap draws over families at seed 20260923. This is the readout study's `raw_M` estimator, in its own spelling, reached through the same packing, the same 1,024-position target eligibility, the same scoring span and the same batch composition as the admitted extraction. A cell's forward pass is that extraction's own `extract_batch`.

**The support.** The 211 assays, 169 families and 26,943 variants that the three Llama-lineage arms share, assay-id digest `81432626b0fb4e711fccea81c0aa786c7d1d69b2dee7f935e2f3aee6ffb282fe`. Each cell replays the eligibility rule from the cohort and then refuses to run unless the result matches the admitted parent extraction assay for assay, including every family label, wild-type identifier, mutation digest and packed-token count.

**The partition.** Every one of the checkpoint's 323 tensors belongs to exactly one of twelve declared groups: `embedding`, `head`, `norm` (all 65 RMSNorm weights, the final one included), `rope` (the 32 stored rotary frequency buffers), and `attention_q1`–`q4` and `mlp_q1`–`q4`, where the quarters are layers 0–7, 8–15, 16–23 and 24–31. The partition is checked to cover the checkpoint rather than assumed to, because that is what makes the two endpoints exact: selecting no group leaves the parent bit-for-bit and selecting every group installs Stage 1 bit-for-bit.

**The waves.** A gate wave first, then a screen, then a confirmation.

| Wave | Support | Cells |
| --- | --- | --- |
| Gate | 211 assays and the 56-family panel | floor and ceiling on each |
| Screen | 56-family panel, digest `5a76dcf59b7b354ff2c54b27694d06cd7ba8fa138d750b36532d3d2d61a5d700` | each of the ten differing groups alone |
| Confirmation | 211 assays | each of the ten differing groups alone, and the complement of each |

The screen panel is a label-blind subsample: sort the eligible assay identifiers, group them by family, sort families as strings, take 56 of them by a NumPy permutation at seed 20260923, then take one assay from each by a seeded integer. No measured effect reaches the choice, and the analysis replays the rule from the eligible support rather than trusting the artefact. It holds 56 assays, 56 families and 7,103 variants, and costs 27.11% of the full support's packed-token work.

**The gate.** The confirmation wave is launched only if the full-support ceiling reproduces Stage 1's admitted +0.14604 [+0.11814, +0.17435] and the floor the parent's −0.00740 [−0.02713, +0.01226], each within the admitted interval. Exact reproduction is the expectation rather than the requirement, because the transplanted model is bit-identical to the checkpoint it reconstructs and the forward is deterministic at batch size one. The analysis refuses to compute any localisation number if either endpoint falls outside its admitted interval: a recovered fraction read against a scale that does not reconstruct the admitted models would mean nothing.

**No cell is selected on an outcome.** The briefed design confirmed only the groups a screen admitted. The measured cost of a full-support cell is 0.755 GPU-hours, which makes confirming all twenty cells affordable, so the confirmation panel is exhaustive over the declared partition and the screen selects nothing. What the screen retains is its two operational roles: it gates the confirmation launch at a quarter of the cost, and it measures the same cells at 56 of 169 resampling units, which is a support-sensitivity check on the confirmation. Its numbers are screening numbers and are never pooled with the confirmation's.

## Checkpoint compatibility and the transplant record

The census is exact and complete over both checkpoints: `results/external_baseline/20260924010633_98d48d6930ac/d2_transplant_census/d2_transplant_census.json`, SHA-256 `ba8318192eee97becc6a5c92fa7c58df7ff9f8c0779ee2b96622cf7d32bf2d0a`, from script SHA-256 `066799fdfbc6e2957560491f16c0d5bfb541b244f0e021acebd5531efffe74aa`. All values below are census counts; sampling intervals do not apply.

The two checkpoints carry the same 323 tensor names, agree on every architectural configuration field compared — model type, hidden and intermediate size, layer, head and key-value head counts, vocabulary size, normalization epsilon, rotary parameters, position budget, activation, bias flags and weight tying — and share one `tokenizer.model`, SHA-256 `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`. Input and output embeddings are untied in both, so an embedding transplant and a head transplant are separate interventions. Every weight tensor is stored float16 in both; the only float32 tensors are the 32 rotary buffers.

| Group | Tensors | Elements | Tensors differing in FP32 | Differing FP32 elements | FP32 bytes identical |
| --- | ---: | ---: | ---: | ---: | --- |
| `embedding` | 1 | 131,072,000 | 1 | 88,160,900 | No |
| `head` | 1 | 131,072,000 | 1 | 127,827,183 | No |
| `norm` | 65 | 266,240 | 0 | 0 | Yes |
| `rope` | 32 | 2,048 | 0 | 0 | Yes |
| `attention_q1` | 32 | 536,870,912 | 32 | 534,121,493 | No |
| `attention_q2` | 32 | 536,870,912 | 32 | 534,882,355 | No |
| `attention_q3` | 32 | 536,870,912 | 32 | 534,612,616 | No |
| `attention_q4` | 32 | 536,870,912 | 32 | 533,822,115 | No |
| `mlp_q1` | 24 | 1,082,130,432 | 24 | 1,076,730,940 | No |
| `mlp_q2` | 24 | 1,082,130,432 | 24 | 1,078,992,257 | No |
| `mlp_q3` | 24 | 1,082,130,432 | 24 | 1,078,643,672 | No |
| `mlp_q4` | 24 | 1,082,130,432 | 24 | 1,076,822,687 | No |

Over the whole checkpoint 6,664,616,218 of 6,738,417,664 stored elements differ, 98.905%. Continued pretraining left every RMSNorm weight and every stored rotary frequency exactly unchanged, so `norm` and `rope` transplants are algebraic no-ops and no cell runs them; their zero contrast would be a consequence of identity rather than evidence that normalization is insensitive to some other intervention. Within the groups that do differ the density varies: 67.26% of the embedding's elements, 97.52% of the head's, and between 98.805% and 99.940% of each individual projection's.

Each cell records its own provenance rather than inheriting it. After the copy, every live checkpoint tensor is digested off the loaded model and matched against the census digest of the checkpoint its group selects, so a cell that did not install exactly what its selection names refuses instead of scoring. The 32 rotary buffers have no destination in the serving transformers version; because the census proves them byte-identical, skipping them is a no-op, and a cell whose skipped tensors were not identical refuses. The full transplant therefore moves 291 tensors and 6,738,415,616 elements, which is every element of the checkpoint except those buffers.

## Retained artefacts

Each cell writes, per assay, the per-variant likelihood delta in its record file and a compressed archive holding the four frozen block summaries at native hidden width — middle and final block output, target-residue-token mean and last residue-token state, mutant minus wild type, with no projection — beside the wild-type states and the same likelihood delta. The endpoint needs only the record file. The archives exist so that a later readout of these transplanted states costs a fit rather than another forward pass; they are the same block indices, 15 and 31, and the same pooling rules as the admitted extraction. No measured effect and no profile score is written by the cell stage: the labels enter only in the analysis.

Re-extracting a cell from scratch costs 0.755 GPU-hours at full support and 0.205 at the screen panel, and every cell is exactly reconstructible from the two checkpoint digests, the census and its declared group specification.
