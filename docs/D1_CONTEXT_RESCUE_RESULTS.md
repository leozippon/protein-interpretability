# Homolog-context mutation rescue: results

Homolog context improves mutation ranking relative to both unrelated and no context for the two ProLLaMA stages on this diagnostic, while remaining below the external profile baseline. ProGen3-3B improves relative to unrelated context but remains worse than no context: its primary contrast is recovery from a larger unrelated-context loss, not rescue of its native baseline. ProteinGLM increases WT likelihood without a resolved mutation-ranking change. These are interface-specific outcomes; they do not establish that evolutionary information was absent from model parameters.

## Admission and measurement

The four scoring cells and their joint analysis completed on H200 on 2026-09-23 under the [frozen context protocol](D1_CONTEXT_RESCUE_PROTOCOL.md). The admitted archive SHA256 is `afc3fe4fecc5c6568705442670540e34840555177f1f4fd2bc1630ebeb3edda3`. Retained evidence is under `results/transfer/d1_followup_20260923/context/`: run `20260923062631_dfd71cd7e171` contains the four score/plan pairs; run `20260923075017_68b04c315654/context_profile_increment` contains `profile_increment.json`, SHA256 `bfb467db60043fd583570f941ea61c4aef1ace37be21892e312d8de648f6ed8a`.

Local admission checked the archive and all five output checksums, all four analysis input hashes, the original cohort hash, complete status, selected/skipped plan accounting, ordered mutation IDs and digests, measured effects and profile scores against the retained cohort. Common-support and stratum counts, saved per-assay contrast arithmetic and undefined counts also reconcile. No model inference or statistical resampling was performed locally. Mutation-level profile residuals and held-family prediction use the same analysis artifact but are reported separately in the [profile-increment results](D1_PROFILE_INCREMENT_RESULTS.md).

Each selected assay retains all 128 mutations in its frozen diagnostic draw. The entire packed input is limited to 1,024 positions without truncation. ProGen3 uses its supported bfloat16 kernel and batch size one selected after the batch-versus-single smoke failed the fixed likelihood tolerance; the ProLLaMA stages use float32 and singleton scoring, and ProteinGLM uses float32 with batch size eight. TF32 is disabled. Retained first-assay batch/single checks pass for all three conditions; singleton comparisons are exact repeat checks, not evidence of full-precision accuracy. Reduced precision remains a ProGen3 limitation.

The later Stage-46 sequence-ID defect does not invalidate these ProGen3 results: the complete retained score file has SHA256 `43adb452af759b0a25a191686aa8980788982425c6c2591fd83c35ffa201d7f4`, explicitly records batch size one, and reports zero repeat differences in each condition. Singleton rows receive the native sequence ID zero. The earlier batched Stage-46 expansion is withdrawn separately in the [canonical audit](INTERPRETABILITY_TRANSFER_AUDIT.md#0-objective-and-current-programme); its semantic defect must not be described merely as reduced-precision drift. Correcting it does not remove the numerical batch disagreements observed in the new readout qualification.

All ranking differences are dimensionless Spearman differences. WT likelihood differences are nats per scored token within a checkpoint, not nats per residue and not a cross-tokenizer magnitude ranking. Estimates first average assays within each 50%-identity WT family, then weight families equally. Brackets are exploratory, unadjusted percentile 95% intervals from 2,000 family-bootstrap resamples. Intervals are withheld below eight families. No context-rescue endpoint is undefined on any retained pooled or stratum support; each reported endpoint uses all assays in its row. Withheld intervals below the unit floor are distinct from undefined correlations.

## Support and exclusions

| Model | Native assays / families / variants | Excluded assays: target too long / no feasible matched context | Exact shared assays / families / variants |
| --- | ---: | ---: | ---: |
| ProGen3-3B | 164 / 134 / 20,992 | 16 / 37 | 164 / 134 / 20,992 |
| ProteinGLM-7B-CLM | 164 / 134 / 20,992 | 16 / 37 | 164 / 134 / 20,992 |
| ProLLaMA Stage 1 | 195 / 158 / 24,960 | 6 / 16 | 164 / 134 / 20,992 |
| ProLLaMA Stage 2 | 195 / 158 / 24,960 | 6 / 16 | 164 / 134 / 20,992 |

The source catalogue has 217 assays. Every exclusion is retained in the corresponding plan and score artifact. The exact common panel matches ordered mutations, WT identity, family, both draw digests, profile scores and measured effects across all four arms. Context sequences remain model-specific because token-count matching and feasibility depend on the tokenizer. Shared mutation support therefore does not imply a shared context intervention. The ProLLaMA common-panel sensitivity removes 31 native-eligible assays per stage.

The separate component diagnostic scores all 217 native-eligible assays, whereas these ProLLaMA context comparisons retain 195. Both no-context implementations declare the same bare `Seq=<...>` residue-token likelihood functional, but perform FP32 summation differently: the component scorer masks the full token run, while the context scorer sums the contiguous scored slice. A retained-output consistency check on all 195 overlapping assays matches mutation digests and experimental effects exactly; maximum absolute mutant-score differences are 0.000732421875 nats for Stage 1 and 0.00048828125 nats for Stage 2, with maximum absolute per-assay Spearman differences 0.0003662333 and 0.0002518353, respectively. These are descriptive finite-artifact maxima, without sampling intervals. They are not a changed declared interface; the records remain separate, and neither their numerical values nor their different-support aggregate contrasts are silently substituted for one another.

## Pooled ranking and WT likelihood

Here H denotes homolog context, U matched unrelated context, and N no context. H−U is primary; H−N and U−N are the prespecified secondary contrasts. “Native = shared” means the complete saved stratum and its summaries are identical on the two supports.

| Model | Support | H−U Δρ | H−N Δρ | U−N Δρ | WT H−U, nats/scored token |
| --- | --- | ---: | ---: | ---: | ---: |
| ProGen3-3B | native = shared | +0.1007 [+0.0616, +0.1378] | -0.1090 [-0.1382, -0.0803] | -0.2096 [-0.2520, -0.1702] | +0.7831 [+0.6777, +0.8869] |
| ProteinGLM-7B-CLM | native = shared | -0.0007 [-0.0155, +0.0133] | +0.0012 [-0.0128, +0.0143] | +0.0019 [-0.0008, +0.0048] | +0.1081 [+0.0663, +0.1568] |
| ProLLaMA Stage 1 | native | +0.1304 [+0.0960, +0.1671] | +0.1336 [+0.1003, +0.1696] | +0.0032 [-0.0065, +0.0134] | +1.8780 [+1.7755, +1.9896] |
| ProLLaMA Stage 1 | shared | +0.1369 [+0.0978, +0.1778] | +0.1390 [+0.1016, +0.1752] | +0.0021 [-0.0097, +0.0142] | +1.9061 [+1.7900, +2.0182] |
| ProLLaMA Stage 2 | native | +0.1186 [+0.0856, +0.1533] | +0.0831 [+0.0495, +0.1172] | -0.0355 [-0.0541, -0.0182] | +1.6990 [+1.5880, +1.8178] |
| ProLLaMA Stage 2 | shared | +0.1253 [+0.0904, +0.1616] | +0.0903 [+0.0535, +0.1284] | -0.0350 [-0.0547, -0.0156] | +1.7601 [+1.6272, +1.8879] |

The ProLLaMA primary gains survive restriction to the exact common mutation panel and exceed their own no-context performance. Stage 2 also loses ranking under unrelated context, so its primary H−U gain combines improvement over N with recovery from that loss. ProGen3’s positive primary contrast must instead be read alongside its negative H−N contrast. ProteinGLM’s positive WT-likelihood contrast and unresolved ranking contrast distinguish the measured endpoints without establishing a zero ranking effect or failure to encode fitness-relevant information.

### MODEL−LOOKUP under each condition

| Model | Support | N−LOOKUP | U−LOOKUP | H−LOOKUP |
| --- | --- | ---: | ---: | ---: |
| ProGen3-3B | native = shared | +0.0101 [-0.0204, +0.0389] | -0.1995 [-0.2488, -0.1542] | -0.0988 [-0.1306, -0.0679] |
| ProteinGLM-7B-CLM | native = shared | +0.0214 [-0.0060, +0.0484] | +0.0233 [-0.0031, +0.0512] | +0.0227 [-0.0017, +0.0466] |
| ProLLaMA Stage 1 | native | -0.2171 [-0.2534, -0.1800] | -0.2139 [-0.2530, -0.1758] | -0.0835 [-0.1095, -0.0583] |
| ProLLaMA Stage 1 | shared | -0.2298 [-0.2742, -0.1885] | -0.2277 [-0.2736, -0.1851] | -0.0908 [-0.1195, -0.0635] |
| ProLLaMA Stage 2 | native | -0.2023 [-0.2365, -0.1679] | -0.2377 [-0.2740, -0.2034] | -0.1192 [-0.1457, -0.0925] |
| ProLLaMA Stage 2 | shared | -0.2134 [-0.2543, -0.1753] | -0.2484 [-0.2913, -0.2065] | -0.1231 [-0.1516, -0.0939] |

Homolog context narrows the ProLLaMA deficit relative to LOOKUP, but does not remove it. ProGen3 remains below LOOKUP under homolog context despite its gain over U. ProteinGLM’s condition-specific differences from LOOKUP remain unresolved on this context-feasible draw. These estimates neither replace historical larger mutation draws nor contradict a historical positive MODEL−LOOKUP on different support.

## Prespecified overlap and identity strata

### Low local overlap

All models retain 21 assays / 19 families / 2,688 variants with selected-homolog longest common substring below 10 residues. Each model’s native and shared summaries are identical in this stratum. This is a selected-fragment condition, not proof of absent distributed homology or a direct-copying mechanism exclusion.

| Model | H−U Δρ | H−N Δρ | U−N Δρ | WT H−U, nats/scored token |
| --- | ---: | ---: | ---: | ---: |
| ProGen3-3B | +0.1694 [+0.1007, +0.2437] | -0.1603 [-0.2724, -0.0474] | -0.3297 [-0.4358, -0.2333] | +0.3473 [+0.2028, +0.5230] |
| ProteinGLM-7B-CLM | +0.0042 [-0.0549, +0.0574] | +0.0098 [-0.0471, +0.0608] | +0.0056 [-0.0050, +0.0174] | +0.2007 [+0.0794, +0.3501] |
| ProLLaMA Stage 1 | +0.1727 [+0.0628, +0.2813] | +0.1713 [+0.0642, +0.2729] | -0.0014 [-0.0351, +0.0305] | +1.2033 [+0.9861, +1.4146] |
| ProLLaMA Stage 2 | +0.1589 [+0.0452, +0.2888] | +0.1530 [+0.0437, +0.2728] | -0.0059 [-0.0425, +0.0324] | +0.8229 [+0.5730, +1.0742] |

| Model | N−LOOKUP | U−LOOKUP | H−LOOKUP |
| --- | ---: | ---: | ---: |
| ProGen3-3B | -0.0044 [-0.1085, +0.0933] | -0.3341 [-0.4606, -0.2153] | -0.1647 [-0.2716, -0.0636] |
| ProteinGLM-7B-CLM | +0.0403 [-0.0412, +0.1343] | +0.0459 [-0.0313, +0.1332] | +0.0501 [-0.0053, +0.1138] |
| ProLLaMA Stage 1 | -0.2564 [-0.3896, -0.1153] | -0.2578 [-0.4019, -0.1181] | -0.0850 [-0.1546, -0.0143] |
| ProLLaMA Stage 2 | -0.2582 [-0.3793, -0.1229] | -0.2641 [-0.4017, -0.1323] | -0.1052 [-0.1819, -0.0254] |

The low-overlap results retain the pooled interpretation: ProLLaMA gains over N, ProGen3 recovers only part of the loss induced by U, and ProteinGLM’s ranking contrasts remain unresolved despite higher WT likelihood. Positive H−U alone therefore does not define rescue of an existing ranking capability.

### Low selected-homolog identity

Identity below 50% is query-normalized identity for the selected homolog fragment. Every row falls below the eight-family interval floor, so only point estimates are reported; none supports a low-identity population claim. Variants equal 128 times the assay count. Different tokenizer-specific selected homologs can put the same mutation panel into different strata.

| Model | Support | Assays / families | H−U Δρ | H−N Δρ | U−N Δρ | WT H−U, nats/scored token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ProGen3-3B | native = shared | 4 / 4 | -0.0428 [withheld] | -0.0711 [withheld] | -0.0283 [withheld] | +0.1253 [withheld] |
| ProteinGLM-7B-CLM | native = shared | 4 / 4 | -0.0060 [withheld] | -0.0050 [withheld] | +0.0010 [withheld] | +0.0074 [withheld] |
| ProLLaMA Stage 1 | native | 6 / 6 | +0.1401 [withheld] | +0.1624 [withheld] | +0.0224 [withheld] | +0.4020 [withheld] |
| ProLLaMA Stage 1 | shared | 3 / 3 | +0.0830 [withheld] | +0.0975 [withheld] | +0.0146 [withheld] | +0.2828 [withheld] |
| ProLLaMA Stage 2 | native | 6 / 6 | +0.0405 [withheld] | +0.0631 [withheld] | +0.0226 [withheld] | +0.2221 [withheld] |
| ProLLaMA Stage 2 | shared | 3 / 3 | +0.0321 [withheld] | +0.0602 [withheld] | +0.0282 [withheld] | +0.2310 [withheld] |

| Model | Support | N−LOOKUP | U−LOOKUP | H−LOOKUP |
| --- | --- | ---: | ---: | ---: |
| ProGen3-3B | native = shared | +0.0494 [withheld] | +0.0212 [withheld] | -0.0217 [withheld] |
| ProteinGLM-7B-CLM | native = shared | +0.0175 [withheld] | +0.0185 [withheld] | +0.0125 [withheld] |
| ProLLaMA Stage 1 | native | -0.1628 [withheld] | -0.1405 [withheld] | -0.0004 [withheld] |
| ProLLaMA Stage 1 | shared | -0.0472 [withheld] | -0.0326 [withheld] | +0.0504 [withheld] |
| ProLLaMA Stage 2 | native | -0.1481 [withheld] | -0.1255 [withheld] | -0.0850 [withheld] |
| ProLLaMA Stage 2 | shared | -0.0735 [withheld] | -0.0454 [withheld] | -0.0133 [withheld] |

### Joint low-identity and low-overlap stratum

Each model has only one assay / one family / 128 variants; native and shared summaries coincide. These are descriptive single-family contrasts with every interval withheld.

| Model | H−U Δρ | H−N Δρ | U−N Δρ | WT H−U, nats/scored token |
| --- | ---: | ---: | ---: | ---: |
| ProGen3-3B | -0.0168 [withheld] | -0.0701 [withheld] | -0.0533 [withheld] | +0.3058 [withheld] |
| ProteinGLM-7B-CLM | -0.0048 [withheld] | +0.0007 [withheld] | +0.0055 [withheld] | -0.0002 [withheld] |
| ProLLaMA Stage 1 | +0.0323 [withheld] | +0.0509 [withheld] | +0.0187 [withheld] | +0.0667 [withheld] |
| ProLLaMA Stage 2 | +0.0059 [withheld] | +0.0484 [withheld] | +0.0426 [withheld] | +0.0771 [withheld] |

| Model | N−LOOKUP | U−LOOKUP | H−LOOKUP |
| --- | ---: | ---: | ---: |
| ProGen3-3B | +0.0665 [withheld] | +0.0132 [withheld] | -0.0036 [withheld] |
| ProteinGLM-7B-CLM | +0.0602 [withheld] | +0.0657 [withheld] | +0.0609 [withheld] |
| ProLLaMA Stage 1 | -0.0012 [withheld] | +0.0175 [withheld] | +0.0497 [withheld] |
| ProLLaMA Stage 2 | -0.0306 [withheld] | +0.0120 [withheld] | +0.0178 [withheld] |

## Interpretation limits

The assay draw, context selection and analyses are exploratory and use previously studied experimental sources. Family resampling addresses dependence within the retained WT clustering, not remote-superfamily disjointness or uncertainty over alternative homolog choices. One fragment per condition is not an MSA; absence from the target hit list is an operational unrelated control, not certified evolutionary nonhomology. No model-universe, training-cause or circuit conclusion follows from differences between these checkpoints.

The strongest explanatory distinction is between a context gain over a harmful control and a gain over the native baseline. The experiment identifies both patterns, plus a positive likelihood effect whose ranking consequence remains unresolved. It does not show that a model’s weights lacked family information, that ranking is completely insensitive to likelihood gains, or that the remaining gap to LOOKUP has a single cause.
