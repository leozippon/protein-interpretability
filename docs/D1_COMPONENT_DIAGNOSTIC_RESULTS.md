# ProLLaMA component diagnostic results

## Admitted checkpoint inventory

The exact tensor inventory completed on H200 on 2026-09-23 from frozen snapshot `20260923072653_3d138ce24e15`. Its purpose is to establish whether the proposed Stage 1/Stage 2 component exchanges change checkpoint values before allocating computation to hybrid models. The [component protocol](D1_GENERATION_DIAGNOSTIC_PROTOCOL.md#component-intervention-follow-up) defines the subsequent measurements and equivalence handling.

The retained artifact is `results/transfer/d1_followup_20260923/component_inventory/component_tensor_inventory.json`, SHA256 `452da6f597f2c5c3b06d9880c7a19a545eb9742100924abbb68cddfeaa802a69`. It contains the two checkpoint shard/index digests, every tensor's shape, native and FP32 digests, exact differing-element counts, component digests and factorial equivalence classes. The inventory script SHA256 is `6f404eb6646bc52a1b694f81906a65203f5415b1012cf6a3ecda98341440d430`. Local admission verified the artifact checksum, reconciled changed tensor/element counts against its tensor rows, and passed the completed artifact through the equivalence validator without model inference or resampling.

All values below are exact census counts over the retained checkpoint tensors; sampling intervals do not apply. “Elements” includes stored buffers where present and is not uniformly a trainable-parameter count.

| Component | Stored tensors | Stored elements | Tensors with differing FP32 values | Differing FP32 elements | Native bytes identical? | FP32 bytes identical? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Embedding | 1 | 131,072,000 | 0 | 0 | Yes | Yes |
| Transformer body, including final norm | 321 | 6,476,273,664 | 224 | 6,403,429,458 | No | No |
| LM head | 1 | 131,072,000 | 0 | 0 | Yes | Yes |

Embedding and head tensors are stored in float16 in both checkpoints; the body contains float16 and float32 tensors in corresponding pairs. Exact FP32 comparison matches the dtype used by the component measurement. Both the embedding and head are identical even before conversion, so their equality is not created by reduced precision during analysis.

Within the body, each of the 32 layers has changed tensors for the attention query, key, value and output projections and the MLP gate, up and down projections: 32 changed tensors in each of these seven categories, 224 in total. The 32 input layer norms, 32 post-attention layer norms, 32 rotary-frequency buffers and final norm are exactly unchanged. This inventories where checkpoint values differ; it does not measure which changed values cause a particular behavior.

## Consequence for the proposed factorial

A combination is ordered as embedding/body/head, with `0` denoting Stage 1 and `1` Stage 2. The inventory proves the following parameter-equivalence classes in the declared FP32 intervention:

| Observed native representative | Parameter-equivalent combinations |
| --- | --- |
| `000` | `000`, `001`, `100`, `101` |
| `111` | `010`, `011`, `110`, `111` |

Thus the proposed factorial contains two distinct parameter settings, represented by the native endpoints. Six further hybrid evaluations would repeat one of those settings. They were not launched, and the equivalent labels are not counted as independently measured cells or additional generation attempts.

Exchanging the embedding or head performs no parameter intervention. Its zero contrast is an algebraic consequence of identity, not empirical evidence that those components are unimportant or insensitive to other interventions. The body marginal and each conditional body contrast reduce to the same Stage 2-minus-Stage 1 endpoint comparison. Aggregating weights by observed representative before resampling preserves that dependence and prevents spuriously narrow intervals from duplicated generation samples.

The tensor inventory therefore limits the original mechanistic design: this pair cannot distinguish an embedding-mediated change from a head-mediated change by swapping those unchanged components, nor provide independent factorial evidence beyond the body-associated endpoint contrast. It does establish that the stored checkpoint differences are confined to the listed Transformer projections. A resulting endpoint difference would not identify a circuit, assign effects among those projections, or isolate the training process responsible.

## Completed native endpoints and attention/MLP refinement

Both native FP32 measurements, the two attention/MLP hybrids and both analyses completed on H200 on 2026-09-23. The admitted archive SHA256 is `adc34cd7718068cb4c32c08690bab8d0e0165ebea24db409eca2c70cbb1f73e5`. Evidence is under `results/transfer/d1_followup_20260923/components/`: run `20260923063207_0c10dded9867/swap_native` contains the native cells; run `20260923075017_68b04c315654` contains `component_contrasts`, `attention_mlp_01`, `attention_mlp_10` and `attention_mlp_contrasts`. The archive and saved output checksums passed; all five coarse-analysis and 18 refinement-analysis source hashes match retained files. Mutation supports, attempt counts, generation outcome counts, stop reasons and reconstruction receipts reconcile. Admission performed no new model inference or statistical resampling.

The native reconstruction maximum absolute logit error was exactly zero for both endpoints in each of the native and hybrid workers. Mutation scoring used singleton FP32 with TF32 disabled; hybrid workers also verified staged checkpoint file hashes against the inventory. Generation used FP32 with the protocol’s fixed residue budget and decoding recipe. These measurements are separate from the earlier bfloat16 generation confirmation.

Intervals below are exploratory, unadjusted percentile 95% intervals from 2,000 resamples. Mutation intervals resample paired wild-type families at 50% identity, averaging assays within family and then families equally. Generation intervals resample the eight matched seed batches of eight attempts jointly across cells; equivalent factorial labels never contribute independent observations. Intervals condition on these checkpoints and the frozen panel, omit training-seed and panel-selection uncertainty, and do not establish equivalence when they include zero.

### Full native mutation support

All 217 native-eligible assays, 174 families and 27,711 retained variants were scored at both endpoints, with no context exclusions or undefined-correlation exclusions. The Stage 2-minus-Stage 1 family-equal Spearman difference is +0.0113 [−0.0023, +0.0266], dimensionless. This diagnostic draw does not resolve the mutation gain motivating the experiment; it is not a replacement for historical estimates on their larger variant draws.

For the same two native cells, all-attempt Pfam recognition changes by −6.25 percentage points [−18.75, +7.81], while native completion changes by +23.44 percentage points [+15.62, +32.81], on 64 attempts per endpoint and eight paired seed batches. Complete-with-Pfam changes by −4.69 percentage points [−14.06, +6.25]; complete-without-Pfam changes by +28.12 percentage points [+14.06, +39.06]. The recognition contrast remains unresolved at this generation sample size. The coarse body marginal and its conditional contrasts equal these endpoint contrasts; embedding/head contrasts are algebraic no-ops, not tests of sensitivity to another possible intervention.

### Frozen four-cell refinement

The [bounded refinement](D1_GENERATION_DIAGNOSTIC_PROTOCOL.md#bounded-attentionmlp-refinement) retains 32 assays from 32 label-blind selected families and all 4,096 original selected variants, with no undefined-correlation exclusions. Panel SHA256 is `4110a38b5aeabbb5d552bbed3f1788adc888681386977b9035e79a51b7d1f15b`. Cell labels here are attention/MLP: `00` and `11` reuse the two native checkpoints, `01` has Stage 1 attention and Stage 2 MLP, and `10` has Stage 2 attention and Stage 1 MLP. Native mutation rows are restricted to this panel; native generation attempts are reused without being counted as new draws.

Each mutation entry below is family-equal Spearman [95% interval] on those 32 families. Generation entries give count/64 followed by percent [95% seed-batch interval], with eight randomization units per cell.

| Cell | Mutation Spearman | Any Pfam | Native complete |
| --- | --- | --- | --- |
| `00` | +0.1224 [+0.0608, +0.1889] | 9/64; 14.06 [7.81, 20.31] | 46/64; 71.88 [64.06, 79.69] |
| `01` | +0.1288 [+0.0655, +0.1944] | 3/64; 4.69 [1.56, 9.38] | 50/64; 78.12 [65.62, 90.62] |
| `10` | +0.1247 [+0.0578, +0.1972] | 10/64; 15.62 [7.81, 23.44] | 49/64; 76.56 [67.15, 87.50] |
| `11` | +0.1285 [+0.0624, +0.1967] | 5/64; 7.81 [0.00, 17.19] | 61/64; 95.31 [90.62, 100.00] |

Complete-with-Pfam counts were 7/64, 0/64, 8/64 and 4/64 for `00`, `01`, `10` and `11`, respectively; these are exact observed counts. The zero count in `01` is not proof of zero event probability: its empirical seed-batch bootstrap is necessarily degenerate for that endpoint. The separate all-attempt Wilson interval is 0.0%–5.7% for 0/64. All four cells had zero empty or format-failure attempts; their incomplete outputs stopped at the residue budget. Neither normal termination nor a Pfam hit establishes biological completeness or function.

All factorial contrasts are retained below. Mutation differences are dimensionless Spearman; generation differences are percentage points, each with its corresponding 95% interval. Conditional effects always replace the named block from Stage 1 to Stage 2 while holding the other block fixed. Marginal effects average the two conditional effects; interaction is `Y11 − Y10 − Y01 + Y00`.

| Contrast | Mutation ΔSpearman | Any-Pfam Δ, percentage points | Completion Δ, percentage points |
| --- | --- | --- | --- |
| Native Stage 2 − Stage 1 | +0.0060 [-0.0201, +0.0340] | -6.25 [-18.75, +7.81] | +23.44 [+15.62, +32.81] |
| Attention marginal | +0.0010 [-0.0138, +0.0178] | +2.34 [-6.25, +10.94] | +10.94 [-2.34, +24.22] |
| MLP marginal | +0.0051 [-0.0119, +0.0218] | -8.59 [-17.19, -0.78] | +12.50 [+3.91, +20.31] |
| Attention, Stage 1 MLP | +0.0022 [-0.0136, +0.0216] | +1.56 [-9.38, +10.94] | +4.69 [-9.38, +20.31] |
| Attention, Stage 2 MLP | -0.0003 [-0.0198, +0.0194] | +3.12 [-6.25, +15.62] | +17.19 [+3.12, +31.25] |
| MLP, Stage 1 attention | +0.0063 [-0.0103, +0.0241] | -9.38 [-17.19, +0.00] | +6.25 [-3.12, +15.62] |
| MLP, Stage 2 attention | +0.0038 [-0.0186, +0.0258] | -7.81 [-20.31, +4.69] | +18.75 [+7.81, +26.56] |
| Attention × MLP interaction | -0.0025 [-0.0236, +0.0173] | +1.56 [-10.94, +15.62] | +12.50 [+1.56, +23.44] |

| Contrast | Complete-with-Pfam Δ, percentage points | Complete-without-Pfam Δ, percentage points |
| --- | --- | --- |
| Native Stage 2 − Stage 1 | -4.69 [-14.06, +6.25] | +28.12 [+14.06, +39.06] |
| Attention marginal | +3.91 [-1.56, +10.16] | +7.03 [-7.81, +21.88] |
| MLP marginal | -8.59 [-15.62, -2.34] | +21.09 [+11.72, +30.47] |
| Attention, Stage 1 MLP | +1.56 [-6.25, +7.81] | +3.12 [-12.50, +18.75] |
| Attention, Stage 2 MLP | +6.25 [+0.00, +15.62] | +10.94 [-3.12, +26.56] |
| MLP, Stage 1 attention | -10.94 [-15.62, -6.25] | +17.19 [+7.81, +26.56] |
| MLP, Stage 2 attention | -6.25 [-17.19, +4.69] | +25.00 [+14.06, +34.41] |
| Attention × MLP interaction | +4.69 [-4.69, +15.62] | +7.81 [+3.12, +14.06] |

The MLP exchange has a negative average effect on Pfam recognition in this factorial: −8.59 percentage points [−17.19, −0.78], accompanied by increased completion and complete-without-Pfam outcomes. Its complete-with-Pfam marginal is also negative, as shown above. Both conditional any-Pfam point estimates are negative, but their intervals include or touch zero; the attention marginal and any-Pfam interaction remain unresolved. Completion shows an interaction, so termination behavior cannot be assigned to one block independently of the other.

No mutation marginal, conditional effect or interaction is resolved on the selected panel, and the native mutation contrast is itself unresolved. Consequently, these results do not establish that mutation improvement and generation loss reside in different blocks, nor that an MLP change explains the mutation gain. They support only an exploratory effect of the specified MLP exchange on this profile-recognition endpoint, alongside altered termination behavior. Cross-stage incompatibility can contribute to a hybrid’s output distribution; the interaction analysis does not eliminate that possibility. Only eight seed batches and 64 attempts per cell support the generation intervals, and multiple unadjusted contrasts were examined. No circuit, training cause, biological function or general loss of generation capability is identified.
