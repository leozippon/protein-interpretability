# Frozen-representation mutation readout: results

**Both stages admitted, 2026-09-23.** The original five-arm study completed and was admitted first; the subsequently authorized full-available-panel expansion then completed and passed final admission at 34 extraction arms and 132 fits. This file reports both, with the five original arms carried inside the expanded tables rather than reported twice. The [protocol](D1_READOUT_PROTOCOL.md) fixes the recipe and the expansion addendum; the [canonical audit](INTERPRETABILITY_TRANSFER_AUDIT.md) remains authoritative for the scientific plan and the limitations catalogue.

The panel separates two readings of a positive representation increment. Adding frozen representations to the equally supervised likelihood/profile/sequence baseline raises held-cluster Spearman above zero with a resolved interval in 3 of the 33 arms on the 201-assay anchor — ProteinGLM-7B-CLM at all three split seeds, ProGen3-3B at two of three, ProGen2-base at one of three — and the largest such anchor point estimate is +0.02255. In the other 30 anchor arms the increment is unresolved or resolved below zero, and all 14 literal-text arms are resolved below zero at all three seeds. On the separate 40-assay function-conditioned panel, at 30 resampling units rather than 163, ProteinGLM reaches +0.05139. Against that, the representation readout ranks mutations better than the native likelihood in 69 of the 99 anchor arm-seed cells. The likelihood is therefore demonstrably not the best readout of the same forward pass, while the information that better readout reaches is, in 31 of 34 arms, already reached by the matched supervised baseline. That pattern constrains the readout explanation much more than it supports it, and it does so without licensing a representation-ceiling claim, because the readout class here is one fixed compressed linear family.

## Evidence and admission

Both stages share one cohort, one fold-generation procedure and one fitting recipe. The original stage used extraction freeze `20260923103322_004fdfab72bf` and analysis freeze `20260923105347_9e1188dbfcda`; its archive has SHA-256 `0f64c869f4f83c98181313b23ae24fa71e2c3653e7ce0f795827fb533a22a53b`, 47,820,558 bytes and 31 files, retained under `results/transfer/readout_20260923/complete/`, with its passed five-arm receipt at `results/transfer/readout_20260923/final_admission.json`.

The expansion is bound by the revised execution roster `results/transfer/readout_expansion_20260923/expected.json`, SHA-256 `1bf26a6ae8d71e3c0f790f12c92da7d3dd4a9add7be6f7ef807836297e12dd0c`, which an execution-only revision derived from parent `e5ac45f8fe36c3e81007a652e9d3e8018d873f821dcc2c88efea7a11b528d21a`, archived unchanged as `expected.v2.json`. That revision altered only batch size and extraction-code hashes for five re-extracted arms; supports, fold plans, feature definitions and statistical code hashes are byte-identical to the parent. The collected archive `results/transfer/readout_expansion_20260923/complete.tar.gz` has SHA-256 `8ccd2053baec6d498de7172ff2d90c19e2a086c29e34eb15f60bf02b74f650da`, 323,535,479 bytes and 178 members, holding 34 canonical manifests and 132 fit reports and no feature archive.

Final local admission returned `status: admitted` in receipt `results/transfer/readout_expansion_20260923/final_admission.json`, SHA-256 `763e7de57b625d55b88bbbe87ad5a68b226693a79fb5d1b5970505aeb6b9f695`, covering 34 extraction receipts, 132 report paths and 17 panels. It reconstructed every nested fold independently, bound each fit to its source manifests and NPZ digests, and compared the matched baseline predictions, tuning and metrics for every exact-support and seed group: all 369 such groups differ by exactly 0.0 at a tolerance of atol = rtol = 1e-8. Re-running the transported checker against the transported roster and extraction receipt reproduces that receipt byte-for-byte, to the same SHA-256. The receipt binds artifact identity, supports, folds and matched-baseline agreement; it makes no statistical or biological claim.

Because the original five-arm outcomes had already been inspected when the expansion was authorized, the expansion is an exploratory extension of an inspected result, not an outcome-blind independent confirmation of it. Its interfaces, eligibility and required fits were frozen in the roster before its outcomes were read.

## Panel, strata and estimation

The 34 admitted arms fall into the three declared interface strata, which are never pooled below.

| Stratum | Arms | Content |
| --- | ---: | --- |
| `native_sequence` | 19 | Residue tokenizers, multi-residue BPE and joint-model protein delimiters in their declared native rendering: ProGen2 five sizes, ProGen3-112M and 3B, ProteinGLM-7B-CLM, ProtGPT2, ProtGPT3-1.3B, RITA-XL, Galactica four sizes, InstructProtein, the Llama-2-7B parent and both ProLLaMA stages |
| `literal_text_AA` | 14 | The same protein sequence fed to a text model as a literal amino-acid string: GPT-2 four sizes, DialoGPT-small, Qwen2.5-0.5B/0.5B-Instruct/7B/32B, Qwen3-8B-Base, Llama-3.2-3B, ByGPT5 three sizes. These are text-model representation controls, not native biological instructions |
| `EC_conditioned` | 1 | ZymCTRL, which receives a frozen genuine wild-type EC annotation on its own conditioned support |

Three panels carry the comparisons. The `anchor201` panel is the unchanged 201 assays, 163 wild-type identity clusters and 25,728 variants of the original study, covered by all 33 unconditioned arms at split seeds 20260923, 20260924 and 20260925. The `EC_conditioned40` panel is 40 EC-annotated assays, 30 clusters and 5,120 variants, carrying ZymCTRL and the five original reference arms at the same three seeds; those five references retain their unconditioned inputs, so matched labels and folds there do not match ZymCTRL's extra EC input. Fifteen `native_*` panels carry the within-arm sensitivities at primary seed 20260923 only, for arms whose own length-eligible support differs from the anchor. Baseline comparison is licensed only inside a panel and seed, where reports share exact assay identifiers; estimates from different panels are not directly comparable model ranks.

Every interval in this file is a pointwise 95% percentile interval from 2,000 paired bootstrap draws at seed 20260923, resampled over the unit *wild-type family at 50% identity* — 163 units on the anchor, 30 on the EC panel, 169, 170 or 172 on the native panels, against a declared floor of 8 units. All 17 panels use that same unit; no report mixes units. Intervals condition on the fitted cross-validation predictions, omit training and tuning variation, and are not adjusted for 34 arms, several endpoints and three seeds. Every reported metric is defined on every assay of its support: the excluded-assay count is 0 in all 1,428 rendered metric rows.

R is the supervised linear readout of four frozen mutant-minus-wild-type summaries — middle and final transformer-block outputs, each pooled by target residue-token mean and by last residue-token state — each projected to 256 fixed Gaussian coordinates for 1,024 R features. B combines the native likelihood M, the external profile score P and 444 sequence descriptors, at 446 features; B+R has 1,470 features before the unpenalized intercept. No pretrained parameter is trained. Spearman values and their differences are dimensionless; rank-MSE differences are in squared standardized-rank units. Token means are token-weighted, not residue-weighted: ByGPT5 literal strings use byte-level states, the other text controls and the Llama lineage use BPE, and the native protein interfaces mix residue tokenizers with multi-residue BPE.

"Cluster-held-out" means the existing single-linkage wild-type groups built from observed DIAMOND matches at 50% identity and 80% coverage of the shorter sequence. It is not certified Pfam-family or CATH-superfamily holdout and does not exclude pretraining overlap. The bounded grouping audit found no repeated ProteinGym entry identifier crossing clusters, nor any crossing among the Swiss-Prot accession mappings available for 193 of the 217 source assays. Remote homology and unmapped entries remain limitations, and checkpoint lineages inside the panel are correlated.

## Native protein and joint interfaces on the anchor panel

All rows use 201 assays, 163 clusters and 25,728 variants. Raw P is +0.36066 [+0.33207, +0.38910] for every arm on this panel; the admission check confirms the shared P, sequence-only and profile-plus-sequence predictions agree to exactly 0.0 across arms on matched support, so those controls are one shared set of values rather than 33 independent ones.

| Model | Raw M | B | R | B+R | Δρ B−M | Δρ (B+R)−M |
| --- | --- | --- | --- | --- | --- | --- |
| Galactica-1.3B | +0.04754 [+0.02567, +0.06915] | +0.43616 [+0.40625, +0.46553] | +0.18900 [+0.16040, +0.21847] | +0.42583 [+0.40018, +0.45241] | +0.38863 [+0.35281, +0.42486] | +0.37829 [+0.34611, +0.41232] |
| Galactica-125M | +0.00328 [-0.01687, +0.02251] | +0.43623 [+0.40668, +0.46556] | +0.17965 [+0.14971, +0.20856] | +0.42399 [+0.39828, +0.44948] | +0.43296 [+0.39820, +0.46929] | +0.42071 [+0.38971, +0.45368] |
| Galactica-30B | +0.13590 [+0.11080, +0.16228] | +0.43742 [+0.40736, +0.46744] | +0.21089 [+0.17935, +0.24158] | +0.42973 [+0.40369, +0.45635] | +0.30152 [+0.26842, +0.33448] | +0.29383 [+0.26372, +0.32433] |
| Galactica-6.7B | +0.08444 [+0.06207, +0.10641] | +0.43656 [+0.40722, +0.46637] | +0.18705 [+0.15384, +0.21786] | +0.42858 [+0.40210, +0.45505] | +0.35212 [+0.31652, +0.38772] | +0.34414 [+0.31172, +0.37769] |
| InstructProtein | +0.28372 [+0.25509, +0.31163] | +0.44892 [+0.41922, +0.47784] | +0.26775 [+0.23929, +0.29681] | +0.43943 [+0.41148, +0.46573] | +0.16520 [+0.13839, +0.19182] | +0.15571 [+0.12770, +0.18332] |
| Llama-2-7B parent | -0.00961 [-0.02917, +0.00957] | +0.43617 [+0.40646, +0.46537] | +0.13215 [+0.10571, +0.15698] | +0.41950 [+0.39211, +0.44713] | +0.44577 [+0.40924, +0.48408] | +0.42911 [+0.39353, +0.46559] |
| ProGen2-base | +0.35634 [+0.32035, +0.39034] | +0.47013 [+0.44054, +0.49942] | +0.35248 [+0.31835, +0.38503] | +0.48354 [+0.45441, +0.51222] | +0.11378 [+0.08895, +0.14120] | +0.12719 [+0.10063, +0.15649] |
| ProGen2-large | +0.36610 [+0.33578, +0.39416] | +0.47260 [+0.44323, +0.50096] | +0.39205 [+0.36207, +0.42040] | +0.48166 [+0.45427, +0.50886] | +0.10650 [+0.08405, +0.12916] | +0.11556 [+0.09411, +0.13877] |
| ProGen2-medium | +0.36678 [+0.33620, +0.39552] | +0.47134 [+0.44172, +0.49919] | +0.33663 [+0.30400, +0.36846] | +0.48115 [+0.45387, +0.50810] | +0.10456 [+0.08196, +0.12726] | +0.11436 [+0.09165, +0.13844] |
| ProGen2-small | +0.31870 [+0.28823, +0.34761] | +0.46027 [+0.43182, +0.48819] | +0.28948 [+0.25800, +0.32242] | +0.47270 [+0.44572, +0.49954] | +0.14157 [+0.11519, +0.16952] | +0.15400 [+0.12905, +0.18151] |
| ProGen2-xlarge | +0.39206 [+0.36438, +0.41839] | +0.48853 [+0.45943, +0.51784] | +0.34919 [+0.31870, +0.37830] | +0.48928 [+0.46167, +0.51553] | +0.09648 [+0.07657, +0.11670] | +0.09722 [+0.07833, +0.11620] |
| ProGen3-112M | +0.24283 [+0.20939, +0.27487] | +0.45257 [+0.42254, +0.48207] | +0.27530 [+0.24429, +0.30704] | +0.45923 [+0.43133, +0.48729] | +0.20973 [+0.17444, +0.24677] | +0.21639 [+0.18220, +0.25189] |
| ProGen3-3B | +0.37866 [+0.35136, +0.40529] | +0.47578 [+0.44699, +0.50511] | +0.39996 [+0.37187, +0.42741] | +0.49315 [+0.46596, +0.52000] | +0.09712 [+0.07658, +0.11771] | +0.11448 [+0.09464, +0.13501] |
| ProLLaMA Stage 1 | +0.14338 [+0.11456, +0.17101] | +0.44021 [+0.41036, +0.46904] | +0.23182 [+0.20061, +0.26328] | +0.43409 [+0.40833, +0.46017] | +0.29683 [+0.25851, +0.33712] | +0.29071 [+0.25702, +0.32665] |
| ProLLaMA Stage 2 | +0.15595 [+0.12674, +0.18482] | +0.44079 [+0.41101, +0.46995] | +0.24102 [+0.21053, +0.27310] | +0.43375 [+0.40534, +0.46126] | +0.28484 [+0.25011, +0.32147] | +0.27780 [+0.24432, +0.31167] |
| ProtGPT2 | +0.33865 [+0.30871, +0.36565] | +0.46468 [+0.43472, +0.49297] | +0.28432 [+0.25243, +0.31749] | +0.47431 [+0.44635, +0.50088] | +0.12604 [+0.10317, +0.15082] | +0.13566 [+0.11239, +0.16078] |
| ProtGPT3-1.3B | +0.29833 [+0.26623, +0.32966] | +0.45371 [+0.42306, +0.48292] | +0.29276 [+0.25940, +0.32455] | +0.45584 [+0.42692, +0.48441] | +0.15537 [+0.12737, +0.18383] | +0.15751 [+0.13020, +0.18600] |
| ProteinGLM-7B-CLM | +0.38984 [+0.36161, +0.41669] | +0.48569 [+0.45550, +0.51519] | +0.43891 [+0.40657, +0.47124] | +0.50540 [+0.47558, +0.53454] | +0.09586 [+0.07642, +0.11471] | +0.11556 [+0.09263, +0.13741] |
| RITA-XL | +0.34912 [+0.32006, +0.37699] | +0.46438 [+0.43544, +0.49267] | +0.34533 [+0.31584, +0.37581] | +0.47210 [+0.44535, +0.49830] | +0.11525 [+0.09156, +0.13931] | +0.12297 [+0.10136, +0.14602] |

Paired Δρ R−M at all three split seeds, with the minimum and maximum point estimate across them.

| Model | 20260923 | 20260924 | 20260925 | Point range |
| --- | --- | --- | --- | --- |
| Galactica-1.3B | +0.14146 [+0.10581, +0.17727] | +0.13896 [+0.10310, +0.17390] | +0.13815 [+0.10232, +0.17488] | +0.13815 to +0.14146 |
| Galactica-125M | +0.17638 [+0.13975, +0.21300] | +0.16894 [+0.13291, +0.20518] | +0.17346 [+0.13662, +0.21083] | +0.16894 to +0.17638 |
| Galactica-30B | +0.07499 [+0.03850, +0.10912] | +0.07289 [+0.03655, +0.10729] | +0.07399 [+0.03845, +0.10720] | +0.07289 to +0.07499 |
| Galactica-6.7B | +0.10260 [+0.06185, +0.14096] | +0.10139 [+0.06188, +0.13968] | +0.10340 [+0.06369, +0.14229] | +0.10139 to +0.10340 |
| InstructProtein | -0.01598 [-0.04815, +0.01832] | -0.02533 [-0.05718, +0.00714] | -0.01904 [-0.05152, +0.01484] | -0.02533 to -0.01598 |
| Llama-2-7B parent | +0.14176 [+0.10604, +0.17546] | +0.15216 [+0.11727, +0.18592] | +0.13674 [+0.09979, +0.17184] | +0.13674 to +0.15216 |
| ProGen2-base | -0.00386 [-0.03736, +0.03093] | -0.00626 [-0.03937, +0.02727] | -0.00998 [-0.04424, +0.02418] | -0.00998 to -0.00386 |
| ProGen2-large | +0.02595 [+0.00284, +0.04984] | +0.02353 [+0.00110, +0.04657] | +0.02414 [+0.00056, +0.04808] | +0.02353 to +0.02595 |
| ProGen2-medium | -0.03015 [-0.05932, +0.00113] | -0.03268 [-0.06310, -0.00190] | -0.02931 [-0.05891, +0.00078] | -0.03268 to -0.02931 |
| ProGen2-small | -0.02922 [-0.06370, +0.00437] | -0.03812 [-0.07057, -0.00593] | -0.03472 [-0.06821, -0.00230] | -0.03812 to -0.02922 |
| ProGen2-xlarge | -0.04286 [-0.07159, -0.01485] | -0.03957 [-0.06767, -0.01204] | -0.04250 [-0.07146, -0.01374] | -0.04286 to -0.03957 |
| ProGen3-112M | +0.03246 [-0.00499, +0.07306] | +0.03471 [-0.00323, +0.07459] | +0.03483 [-0.00345, +0.07434] | +0.03246 to +0.03483 |
| ProGen3-3B | +0.02129 [-0.00002, +0.04286] | +0.01599 [-0.00520, +0.03844] | +0.02038 [-0.00222, +0.04358] | +0.01599 to +0.02129 |
| ProLLaMA Stage 1 | +0.08844 [+0.05311, +0.12389] | +0.08429 [+0.04732, +0.11949] | +0.07727 [+0.04058, +0.11248] | +0.07727 to +0.08844 |
| ProLLaMA Stage 2 | +0.08507 [+0.05315, +0.11716] | +0.08320 [+0.05115, +0.11538] | +0.08047 [+0.04809, +0.11327] | +0.08047 to +0.08507 |
| ProtGPT2 | -0.05433 [-0.08731, -0.02103] | -0.05786 [-0.09155, -0.02549] | -0.05582 [-0.08753, -0.02381] | -0.05786 to -0.05433 |
| ProtGPT3-1.3B | -0.00557 [-0.03891, +0.02785] | +0.00242 [-0.02804, +0.03335] | +0.00995 [-0.02208, +0.04357] | -0.00557 to +0.00995 |
| ProteinGLM-7B-CLM | +0.04907 [+0.02213, +0.07606] | +0.05219 [+0.02599, +0.07886] | +0.05147 [+0.02685, +0.07717] | +0.04907 to +0.05219 |
| RITA-XL | -0.00379 [-0.02813, +0.02341] | -0.00577 [-0.03181, +0.02117] | +0.00288 [-0.02334, +0.02944] | -0.00577 to +0.00288 |

The primary endpoint, paired Δρ (B+R)−B, at all three split seeds.

| Model | 20260923 | 20260924 | 20260925 | Point range |
| --- | --- | --- | --- | --- |
| Galactica-1.3B | -0.01033 [-0.02328, +0.00282] | -0.01169 [-0.02442, +0.00123] | -0.01046 [-0.02287, +0.00239] | -0.01169 to -0.01033 |
| Galactica-125M | -0.01225 [-0.02510, +0.00041] | -0.01710 [-0.02895, -0.00609] | -0.01685 [-0.02868, -0.00472] | -0.01710 to -0.01225 |
| Galactica-30B | -0.00769 [-0.02192, +0.00623] | -0.01629 [-0.03235, -0.00137] | -0.01378 [-0.02870, -0.00013] | -0.01629 to -0.00769 |
| Galactica-6.7B | -0.00798 [-0.02044, +0.00409] | -0.01271 [-0.02526, -0.00072] | -0.00986 [-0.02243, +0.00241] | -0.01271 to -0.00798 |
| InstructProtein | -0.00949 [-0.02206, +0.00277] | -0.01646 [-0.02872, -0.00463] | -0.01551 [-0.02906, -0.00213] | -0.01646 to -0.00949 |
| Llama-2-7B parent | -0.01667 [-0.02741, -0.00699] | -0.01598 [-0.02710, -0.00639] | -0.02064 [-0.03176, -0.01104] | -0.02064 to -0.01598 |
| ProGen2-base | +0.01341 [+0.00099, +0.02617] | +0.00278 [-0.00771, +0.01362] | +0.00270 [-0.00890, +0.01501] | +0.00270 to +0.01341 |
| ProGen2-large | +0.00906 [-0.00382, +0.02148] | +0.00256 [-0.00772, +0.01239] | +0.00335 [-0.00801, +0.01472] | +0.00256 to +0.00906 |
| ProGen2-medium | +0.00981 [-0.00194, +0.02132] | +0.00274 [-0.00795, +0.01237] | +0.00446 [-0.00564, +0.01463] | +0.00274 to +0.00981 |
| ProGen2-small | +0.01243 [-0.00025, +0.02515] | +0.00756 [-0.00382, +0.01884] | +0.00566 [-0.00610, +0.01748] | +0.00566 to +0.01243 |
| ProGen2-xlarge | +0.00075 [-0.00688, +0.00827] | +0.00199 [-0.00585, +0.01014] | +0.00647 [-0.00206, +0.01511] | +0.00075 to +0.00647 |
| ProGen3-112M | +0.00666 [-0.00642, +0.01965] | +0.00261 [-0.00971, +0.01522] | +0.00313 [-0.00912, +0.01586] | +0.00261 to +0.00666 |
| ProGen3-3B | +0.01736 [+0.00667, +0.02834] | +0.00739 [-0.00154, +0.01636] | +0.01091 [+0.00088, +0.02090] | +0.00739 to +0.01736 |
| ProLLaMA Stage 1 | -0.00612 [-0.02110, +0.00837] | -0.01175 [-0.02658, +0.00271] | -0.01562 [-0.03021, -0.00108] | -0.01562 to -0.00612 |
| ProLLaMA Stage 2 | -0.00704 [-0.02193, +0.00765] | -0.01018 [-0.02432, +0.00453] | -0.01460 [-0.02951, -0.00037] | -0.01460 to -0.00704 |
| ProtGPT2 | +0.00962 [-0.00199, +0.02141] | +0.00355 [-0.00772, +0.01478] | +0.00155 [-0.01050, +0.01378] | +0.00155 to +0.00962 |
| ProtGPT3-1.3B | +0.00214 [-0.00977, +0.01421] | -0.00251 [-0.01282, +0.00788] | +0.00324 [-0.00844, +0.01535] | -0.00251 to +0.00324 |
| ProteinGLM-7B-CLM | +0.01970 [+0.00942, +0.02971] | +0.02255 [+0.01196, +0.03301] | +0.02155 [+0.01023, +0.03211] | +0.01970 to +0.02255 |
| RITA-XL | +0.00772 [-0.00201, +0.01785] | +0.00169 [-0.00710, +0.01041] | +0.00352 [-0.00641, +0.01357] | +0.00169 to +0.00772 |

Nine of the nineteen arms have a resolved positive R−M at the primary seed, two are resolved negative — ProGen2-xlarge −0.04286 [−0.07159, −0.01485] and ProtGPT2 −0.05433 [−0.08731, −0.02103] — and eight are unresolved. The sign and size of R−M track how well the native likelihood already ranks mutations. The four Galactica sizes and the three Llama-lineage arms, whose raw M runs from −0.00961 to +0.13590, gain +0.07499 to +0.17638 by reading representations instead. The five ProGen2 sizes and ProtGPT2, whose raw M runs from +0.31870 to +0.39206, change by −0.05433 to +0.02595, with only ProGen2-large resolved above zero.

The primary increment over the matched supervised baseline is much smaller and mostly unresolved. Across the 57 native arm-seed cells, 6 are resolved above zero, 12 resolved below and 39 unresolved. ProteinGLM-7B-CLM is the only arm resolved above zero at all three seeds, spanning +0.01970 to +0.02255; ProGen3-3B is resolved at seeds 20260923 and 20260925 and unresolved at 20260924, spanning +0.00739 to +0.01736; ProGen2-base is resolved only at the primary seed, spanning +0.00270 to +0.01341. The Llama-2-7B parent is resolved below zero at all three seeds, and both ProLLaMA stages turn resolved-negative at seed 20260925 only. The five original arms' values in these tables are the admitted original fits, unchanged.

## Literal amino-acid strings in text models

These 14 arms score the same protein sequences as literal text. They are controls on the readout class and on the descriptor baseline, not measurements of a protein capability.

| Model | Raw M | B | R | B+R | Δρ B−M | Δρ (B+R)−M |
| --- | --- | --- | --- | --- | --- | --- |
| ByGPT5-base-en | -0.00326 [-0.02325, +0.01804] | +0.43580 [+0.40643, +0.46492] | +0.14302 [+0.11547, +0.17167] | +0.41454 [+0.38788, +0.44061] | +0.43906 [+0.40069, +0.47550] | +0.41779 [+0.38186, +0.45210] |
| ByGPT5-medium-en | -0.02027 [-0.04147, +0.00086] | +0.43612 [+0.40678, +0.46536] | +0.14874 [+0.12174, +0.17739] | +0.41953 [+0.39328, +0.44544] | +0.45639 [+0.41737, +0.49514] | +0.43980 [+0.40554, +0.47580] |
| ByGPT5-small-en | -0.01039 [-0.02865, +0.00705] | +0.43596 [+0.40651, +0.46519] | +0.15505 [+0.12653, +0.18218] | +0.41908 [+0.39276, +0.44401] | +0.44635 [+0.41142, +0.48197] | +0.42947 [+0.39730, +0.46198] |
| DialoGPT-small | -0.00808 [-0.03108, +0.01323] | +0.43569 [+0.40634, +0.46489] | +0.06753 [+0.04646, +0.08833] | +0.40521 [+0.37809, +0.43156] | +0.44377 [+0.40522, +0.48331] | +0.41329 [+0.37667, +0.45067] |
| GPT-2 | -0.00436 [-0.02285, +0.01478] | +0.43626 [+0.40662, +0.46570] | +0.11486 [+0.09089, +0.13927] | +0.41032 [+0.38306, +0.43741] | +0.44062 [+0.40563, +0.47721] | +0.41468 [+0.38148, +0.44910] |
| GPT-2-XL | -0.00688 [-0.02896, +0.01501] | +0.43607 [+0.40658, +0.46529] | +0.11654 [+0.09252, +0.14092] | +0.41524 [+0.38984, +0.44066] | +0.44296 [+0.40667, +0.47999] | +0.42213 [+0.38977, +0.45440] |
| GPT-2-large | -0.01597 [-0.03564, +0.00493] | +0.43685 [+0.40740, +0.46598] | +0.12034 [+0.09401, +0.14546] | +0.40791 [+0.38093, +0.43453] | +0.45282 [+0.41484, +0.49214] | +0.42388 [+0.38736, +0.46024] |
| GPT-2-medium | -0.00333 [-0.02165, +0.01484] | +0.43607 [+0.40648, +0.46536] | +0.10193 [+0.07728, +0.12723] | +0.41139 [+0.38470, +0.43817] | +0.43941 [+0.40492, +0.47467] | +0.41472 [+0.38312, +0.44556] |
| Llama-3.2-3B | -0.00086 [-0.02197, +0.01982] | +0.43629 [+0.40679, +0.46556] | +0.08669 [+0.06385, +0.10896] | +0.41510 [+0.38942, +0.44083] | +0.43715 [+0.39903, +0.47663] | +0.41595 [+0.38105, +0.45180] |
| Qwen2.5-0.5B | +0.01770 [-0.00339, +0.03626] | +0.43598 [+0.40624, +0.46517] | +0.09212 [+0.06664, +0.11787] | +0.41537 [+0.38929, +0.44181] | +0.41828 [+0.38219, +0.45586] | +0.39768 [+0.36336, +0.43270] |
| Qwen2.5-0.5B-Instruct | +0.00176 [-0.01694, +0.01878] | +0.43611 [+0.40660, +0.46539] | +0.07659 [+0.05289, +0.10111] | +0.41377 [+0.38613, +0.44099] | +0.43434 [+0.39953, +0.47034] | +0.41201 [+0.37992, +0.44546] |
| Qwen2.5-32B | +0.03052 [+0.00752, +0.05275] | +0.43618 [+0.40659, +0.46531] | +0.15596 [+0.12893, +0.18152] | +0.42503 [+0.39957, +0.45035] | +0.40567 [+0.36700, +0.44510] | +0.39451 [+0.35834, +0.42925] |
| Qwen2.5-7B | +0.00350 [-0.01494, +0.02159] | +0.43602 [+0.40647, +0.46523] | +0.13133 [+0.10545, +0.15751] | +0.41845 [+0.39124, +0.44402] | +0.43252 [+0.39703, +0.46929] | +0.41495 [+0.38257, +0.44828] |
| Qwen3-8B-Base | -0.00608 [-0.02740, +0.01541] | +0.43658 [+0.40706, +0.46565] | +0.09527 [+0.07392, +0.11567] | +0.41094 [+0.38550, +0.43563] | +0.44266 [+0.41020, +0.47625] | +0.41702 [+0.38792, +0.44675] |

Paired Δρ R−M at all three split seeds.

| Model | 20260923 | 20260924 | 20260925 | Point range |
| --- | --- | --- | --- | --- |
| ByGPT5-base-en | +0.14628 [+0.10825, +0.18294] | +0.15805 [+0.12095, +0.19624] | +0.14855 [+0.11142, +0.18560] | +0.14628 to +0.15805 |
| ByGPT5-medium-en | +0.16901 [+0.13135, +0.20805] | +0.16654 [+0.13005, +0.20521] | +0.16586 [+0.12723, +0.20548] | +0.16586 to +0.16901 |
| ByGPT5-small-en | +0.16543 [+0.13049, +0.19938] | +0.16710 [+0.13227, +0.20039] | +0.17027 [+0.13551, +0.20304] | +0.16543 to +0.17027 |
| DialoGPT-small | +0.07561 [+0.04531, +0.10701] | +0.07337 [+0.04337, +0.10379] | +0.07989 [+0.05168, +0.10932] | +0.07337 to +0.07989 |
| GPT-2 | +0.11922 [+0.08818, +0.15191] | +0.12211 [+0.09015, +0.15498] | +0.12779 [+0.09610, +0.16083] | +0.11922 to +0.12779 |
| GPT-2-XL | +0.12343 [+0.08963, +0.15536] | +0.11926 [+0.08702, +0.15062] | +0.11745 [+0.08340, +0.15043] | +0.11745 to +0.12343 |
| GPT-2-large | +0.13631 [+0.09985, +0.17191] | +0.13622 [+0.10061, +0.17107] | +0.13767 [+0.10111, +0.17399] | +0.13622 to +0.13767 |
| GPT-2-medium | +0.10526 [+0.07245, +0.13657] | +0.11243 [+0.08060, +0.14250] | +0.09926 [+0.06634, +0.13010] | +0.09926 to +0.11243 |
| Llama-3.2-3B | +0.08755 [+0.05762, +0.11654] | +0.08588 [+0.05614, +0.11539] | +0.08240 [+0.05477, +0.10932] | +0.08240 to +0.08755 |
| Qwen2.5-0.5B | +0.07442 [+0.04434, +0.10702] | +0.06975 [+0.03782, +0.10205] | +0.07989 [+0.04808, +0.11211] | +0.06975 to +0.07989 |
| Qwen2.5-0.5B-Instruct | +0.07483 [+0.04592, +0.10652] | +0.08423 [+0.05376, +0.11593] | +0.07061 [+0.04059, +0.10089] | +0.07061 to +0.08423 |
| Qwen2.5-32B | +0.12544 [+0.08859, +0.16101] | +0.12493 [+0.08708, +0.16103] | +0.12806 [+0.09117, +0.16423] | +0.12493 to +0.12806 |
| Qwen2.5-7B | +0.12783 [+0.09490, +0.15999] | +0.13451 [+0.10020, +0.16754] | +0.13429 [+0.10066, +0.16791] | +0.12783 to +0.13451 |
| Qwen3-8B-Base | +0.10134 [+0.06808, +0.13424] | +0.10081 [+0.06836, +0.13301] | +0.08931 [+0.05645, +0.12222] | +0.08931 to +0.10134 |

The primary endpoint at all three split seeds.

| Model | 20260923 | 20260924 | 20260925 | Point range |
| --- | --- | --- | --- | --- |
| ByGPT5-base-en | -0.02127 [-0.03504, -0.00760] | -0.02147 [-0.03434, -0.00896] | -0.02768 [-0.04095, -0.01508] | -0.02768 to -0.02127 |
| ByGPT5-medium-en | -0.01659 [-0.02907, -0.00516] | -0.02184 [-0.03520, -0.00965] | -0.02218 [-0.03366, -0.01153] | -0.02218 to -0.01659 |
| ByGPT5-small-en | -0.01688 [-0.02897, -0.00552] | -0.02356 [-0.03560, -0.01233] | -0.02467 [-0.03757, -0.01304] | -0.02467 to -0.01688 |
| DialoGPT-small | -0.03048 [-0.04058, -0.02063] | -0.03283 [-0.04361, -0.02275] | -0.03313 [-0.04258, -0.02352] | -0.03313 to -0.03048 |
| GPT-2 | -0.02594 [-0.03817, -0.01459] | -0.03081 [-0.04310, -0.01885] | -0.02798 [-0.04008, -0.01694] | -0.03081 to -0.02594 |
| GPT-2-XL | -0.02083 [-0.03189, -0.00977] | -0.02895 [-0.04073, -0.01753] | -0.02970 [-0.04118, -0.01909] | -0.02970 to -0.02083 |
| GPT-2-large | -0.02894 [-0.04141, -0.01743] | -0.03573 [-0.04908, -0.02360] | -0.03403 [-0.04632, -0.02304] | -0.03573 to -0.02894 |
| GPT-2-medium | -0.02468 [-0.03685, -0.01355] | -0.02651 [-0.03707, -0.01577] | -0.03110 [-0.04202, -0.02081] | -0.03110 to -0.02468 |
| Llama-3.2-3B | -0.02119 [-0.03213, -0.01079] | -0.02633 [-0.03670, -0.01638] | -0.02878 [-0.03972, -0.01897] | -0.02878 to -0.02119 |
| Qwen2.5-0.5B | -0.02060 [-0.03086, -0.01089] | -0.02389 [-0.03365, -0.01434] | -0.02811 [-0.03760, -0.01916] | -0.02811 to -0.02060 |
| Qwen2.5-0.5B-Instruct | -0.02234 [-0.03170, -0.01318] | -0.02540 [-0.03419, -0.01684] | -0.02921 [-0.03919, -0.02053] | -0.02921 to -0.02234 |
| Qwen2.5-32B | -0.01115 [-0.02257, -0.00055] | -0.01891 [-0.02980, -0.00833] | -0.01590 [-0.02736, -0.00438] | -0.01891 to -0.01115 |
| Qwen2.5-7B | -0.01757 [-0.02935, -0.00599] | -0.01947 [-0.03012, -0.00922] | -0.02393 [-0.03566, -0.01231] | -0.02393 to -0.01757 |
| Qwen3-8B-Base | -0.02564 [-0.03757, -0.01537] | -0.03002 [-0.04050, -0.02048] | -0.03360 [-0.04526, -0.02310] | -0.03360 to -0.02564 |

Raw M is unresolved in 13 of the 14 arms, with points from −0.02027 to +0.01770; only Qwen2.5-32B is resolved above zero, at +0.03052 [+0.00752, +0.05275]. R nonetheless reaches +0.06753 to +0.15596, so R−M is resolved positive in all 42 arm-seed cells, from +0.06975 to +0.17027. This is the clearest demonstration in the panel that a positive R−M is not by itself evidence of protein-specific representational content: a frozen text checkpoint whose likelihood carries no resolved ranking signal still supports a supervised linear readout worth +0.07 to +0.17 Spearman, because R is fitted on experimental labels from other clusters and M is not.

The increment over the matched baseline is resolved below zero in all 42 cells, from −0.01115 to −0.03573, and B itself is nearly constant across these arms at +0.43569 to +0.43685 — the value the shared profile and sequence descriptors reach when M contributes nothing. Adding 1,024 representation coordinates to that baseline consistently costs ranking performance under the fixed ridge grid. Nothing in this stratum is consistent with a readout problem hiding accessible mutation-effect information in text-model states; the whole of what these representations supply is already supplied by profile and sequence descriptors.

## Rank-MSE, the paired secondary endpoint

Positive values mean the baseline-plus-representation predictor has the smaller weighted mean squared error on standardized ranks. Spearman ordering and rank-MSE are separate endpoints, so a negative ranking increment can coexist with a positive but unresolved MSE reduction.

| Model | Stratum | Primary 20260923 | Point range |
| --- | --- | --- | --- |
| Galactica-1.3B | Native sequence | +0.05110 [-0.01256, +0.17520] | +0.03580 to +0.05110 |
| Galactica-125M | Native sequence | +0.04414 [-0.01702, +0.16589] | +0.02868 to +0.04414 |
| Galactica-30B | Native sequence | +0.05346 [-0.01348, +0.17913] | +0.03196 to +0.05346 |
| Galactica-6.7B | Native sequence | +0.05299 [-0.01126, +0.17877] | +0.03805 to +0.05299 |
| InstructProtein | Native sequence | +0.03809 [-0.01779, +0.14350] | +0.00768 to +0.03809 |
| Llama-2-7B parent | Native sequence | +0.03105 [-0.02701, +0.15131] | +0.01698 to +0.03679 |
| ProGen2-base | Native sequence | +0.04766 [-0.00426, +0.14614] | +0.01364 to +0.04766 |
| ProGen2-large | Native sequence | +0.04094 [-0.01174, +0.13968] | +0.00795 to +0.04094 |
| ProGen2-medium | Native sequence | +0.04212 [-0.01007, +0.14311] | +0.01665 to +0.04212 |
| ProGen2-small | Native sequence | +0.05089 [-0.01349, +0.17219] | +0.02943 to +0.05089 |
| ProGen2-xlarge | Native sequence | +0.00767 [-0.03851, +0.09165] | -0.00516 to +0.00767 |
| ProGen3-112M | Native sequence | +0.05760 [-0.00511, +0.17827] | +0.03946 to +0.05760 |
| ProGen3-3B | Native sequence | +0.04312 [-0.00561, +0.13720] | +0.01364 to +0.04312 |
| ProLLaMA Stage 1 | Native sequence | +0.05163 [-0.01431, +0.18027] | +0.02815 to +0.05163 |
| ProLLaMA Stage 2 | Native sequence | +0.04861 [-0.01795, +0.17949] | +0.02863 to +0.04861 |
| ProtGPT2 | Native sequence | +0.04751 [-0.00270, +0.14540] | +0.02458 to +0.04751 |
| ProtGPT3-1.3B | Native sequence | +0.05133 [-0.01051, +0.17265] | +0.03282 to +0.05133 |
| ProteinGLM-7B-CLM | Native sequence | +0.03309 [-0.01563, +0.12301] | +0.02412 to +0.03309 |
| RITA-XL | Native sequence | +0.03398 [-0.01595, +0.12903] | +0.01017 to +0.03398 |
| ByGPT5-base-en | Literal text AA | +0.03112 [-0.02610, +0.14191] | +0.01492 to +0.03417 |
| ByGPT5-medium-en | Literal text AA | +0.03757 [-0.02153, +0.15430] | +0.02201 to +0.03757 |
| ByGPT5-small-en | Literal text AA | +0.03624 [-0.02430, +0.15785] | +0.01827 to +0.03624 |
| DialoGPT-small | Literal text AA | +0.02117 [-0.03455, +0.13724] | +0.00545 to +0.02182 |
| GPT-2 | Literal text AA | -0.03233 [-0.04956, -0.01712] | -0.03233 to +0.02683 |
| GPT-2-XL | Literal text AA | +0.02829 [-0.02730, +0.13955] | +0.00995 to +0.02829 |
| GPT-2-large | Literal text AA | +0.01963 [-0.04587, +0.13878] | +0.01295 to +0.02196 |
| GPT-2-medium | Literal text AA | +0.02392 [-0.03136, +0.13720] | +0.01065 to +0.03020 |
| Llama-3.2-3B | Literal text AA | +0.03208 [-0.02869, +0.14869] | +0.01268 to +0.03208 |
| Qwen2.5-0.5B | Literal text AA | +0.03336 [-0.02882, +0.15638] | +0.01151 to +0.03336 |
| Qwen2.5-0.5B-Instruct | Literal text AA | -0.01204 [-0.03332, +0.01993] | -0.01204 to +0.02513 |
| Qwen2.5-32B | Literal text AA | +0.04025 [-0.02113, +0.15926] | +0.02232 to +0.04025 |
| Qwen2.5-7B | Literal text AA | +0.03942 [-0.02294, +0.16192] | +0.01793 to +0.03942 |
| Qwen3-8B-Base | Literal text AA | +0.03130 [-0.03118, +0.15231] | +0.01162 to +0.03130 |

Thirty-two of the 33 anchor arms have an unresolved MSE reduction at the primary seed; GPT-2 alone is resolved, at −0.03233 [−0.04956, −0.01712]. Over all 99 anchor cells exactly two are resolved, that one and ProGen3-112M at seed 20260924 at +0.05413 [+0.00043, +0.15446]. All 15 native-support MSE reductions are unresolved. On the EC panel, by contrast, 13 of 18 are resolved below zero. Positive point estimates without resolved intervals do not establish better calibrated rank predictions, and the two endpoints need not agree in sign.

## Function-conditioned readout: the EC panel

ZymCTRL receives the lexicographically first complete four-field EC from the frozen genuine-annotation sidecar, SHA-256 `f66015b7710ec008c74b0c76cf61d965e6c4c04cf6f133b9d7b849d364147ab5`, with the identical condition applied to the wild type and every mutant. Exact wild-type matching supplied 44 source assays in 34 clusters; native token-length qualification retained 40 assays in 30 clusters. Multiple EC annotations occur, including for CP2C9, NUD15 and PTEN, and the selected EC is not asserted to identify uniquely the activity that a given DMS assay measures. The five reference arms here are unconditioned, so this panel is an additional-information stratum: it matches labels, supports and folds but not inputs.

| Model | Stratum | Raw M | B | R | B+R |
| --- | --- | --- | --- | --- | --- |
| Llama-2-7B parent | Native sequence | +0.01879 [-0.01441, +0.05325] | +0.35696 [+0.30716, +0.39930] | +0.07546 [+0.03943, +0.10926] | +0.34086 [+0.29625, +0.38313] |
| ProGen3-3B | Native sequence | +0.39474 [+0.33789, +0.44820] | +0.41063 [+0.35793, +0.45977] | +0.39837 [+0.34212, +0.45532] | +0.41370 [+0.36188, +0.46475] |
| ProLLaMA Stage 1 | Native sequence | +0.12654 [+0.07899, +0.17553] | +0.35744 [+0.30864, +0.39973] | +0.13925 [+0.08824, +0.18667] | +0.33704 [+0.29456, +0.37490] |
| ProLLaMA Stage 2 | Native sequence | +0.12724 [+0.07992, +0.17818] | +0.35912 [+0.30995, +0.40154] | +0.18018 [+0.13621, +0.22092] | +0.34710 [+0.30553, +0.38535] |
| ProteinGLM-7B-CLM | Native sequence | +0.39631 [+0.33632, +0.45435] | +0.41438 [+0.35589, +0.46977] | +0.43828 [+0.38585, +0.48940] | +0.45980 [+0.40586, +0.51489] |
| ZymCTRL | EC conditioned | +0.32464 [+0.26974, +0.38025] | +0.38490 [+0.33218, +0.43380] | +0.29441 [+0.25042, +0.33551] | +0.38080 [+0.33678, +0.42369] |

Paired Δρ R−M at all three split seeds.

| Model | 20260923 | 20260924 | 20260925 | Point range |
| --- | --- | --- | --- | --- |
| Llama-2-7B parent | +0.05667 [+0.00135, +0.11119] | +0.06449 [+0.01502, +0.11435] | +0.07452 [+0.02008, +0.12823] | +0.05667 to +0.07452 |
| ProGen3-3B | +0.00364 [-0.02368, +0.03483] | -0.00088 [-0.02948, +0.03047] | +0.00042 [-0.02818, +0.03122] | -0.00088 to +0.00364 |
| ProLLaMA Stage 1 | +0.01271 [-0.02994, +0.05897] | +0.00886 [-0.03348, +0.05241] | +0.01504 [-0.02352, +0.05604] | +0.00886 to +0.01504 |
| ProLLaMA Stage 2 | +0.05293 [+0.01741, +0.09076] | +0.05063 [+0.01406, +0.08726] | +0.03878 [+0.00455, +0.07136] | +0.03878 to +0.05293 |
| ProteinGLM-7B-CLM | +0.04196 [+0.00249, +0.07762] | +0.05242 [+0.01401, +0.08743] | +0.04904 [+0.01249, +0.08258] | +0.04196 to +0.05242 |
| ZymCTRL | -0.03024 [-0.07850, +0.01789] | -0.03180 [-0.08157, +0.01734] | -0.03250 [-0.07794, +0.01110] | -0.03250 to -0.03024 |

The primary endpoint at all three split seeds.

| Model | 20260923 | 20260924 | 20260925 | Point range |
| --- | --- | --- | --- | --- |
| Llama-2-7B parent | -0.01610 [-0.03209, -0.00157] | -0.02924 [-0.04735, -0.01212] | -0.02215 [-0.04061, -0.00575] | -0.02924 to -0.01610 |
| ProGen3-3B | +0.00307 [-0.01353, +0.02065] | +0.01516 [-0.00884, +0.04116] | +0.01799 [-0.00406, +0.04110] | +0.00307 to +0.01799 |
| ProLLaMA Stage 1 | -0.02040 [-0.03789, -0.00566] | -0.03210 [-0.05064, -0.01479] | -0.03361 [-0.05384, -0.01569] | -0.03361 to -0.02040 |
| ProLLaMA Stage 2 | -0.01202 [-0.02819, +0.00295] | -0.01618 [-0.03533, +0.00327] | -0.02560 [-0.04437, -0.00743] | -0.02560 to -0.01202 |
| ProteinGLM-7B-CLM | +0.04542 [+0.02084, +0.06841] | +0.05139 [+0.02511, +0.07745] | +0.04648 [+0.02150, +0.07045] | +0.04542 to +0.05139 |
| ZymCTRL | -0.00410 [-0.02479, +0.01824] | -0.00714 [-0.02735, +0.01610] | -0.00978 [-0.02964, +0.01014] | -0.00978 to -0.00410 |

The paired rank-MSE reduction at all three split seeds.

| Model | 20260923 | 20260924 | 20260925 | Point range |
| --- | --- | --- | --- | --- |
| Llama-2-7B parent | -0.02016 [-0.03855, -0.00639] | -0.03825 [-0.06402, -0.01636] | -0.04546 [-0.07414, -0.01801] | -0.04546 to -0.02016 |
| ProGen3-3B | -0.01304 [-0.03494, +0.00983] | -0.02334 [-0.04682, -0.00105] | -0.02672 [-0.04886, -0.00472] | -0.02672 to -0.01304 |
| ProLLaMA Stage 1 | -0.01327 [-0.02422, -0.00380] | -0.04433 [-0.06261, -0.02685] | -0.04209 [-0.05875, -0.02641] | -0.04433 to -0.01327 |
| ProLLaMA Stage 2 | -0.01044 [-0.02138, +0.00133] | -0.03057 [-0.04528, -0.01654] | -0.03340 [-0.04790, -0.01948] | -0.03340 to -0.01044 |
| ProteinGLM-7B-CLM | -0.02629 [-0.07150, +0.00790] | -0.00602 [-0.03220, +0.02165] | -0.00913 [-0.03535, +0.01650] | -0.02629 to -0.00602 |
| ZymCTRL | -0.01915 [-0.03873, -0.00223] | -0.02342 [-0.04471, -0.00305] | -0.03047 [-0.05128, -0.01194] | -0.03047 to -0.01915 |

ZymCTRL's conditioned native likelihood reaches +0.32464 [+0.26974, +0.38025], below ProGen3-3B's +0.39474 and ProteinGLM's +0.39631 on the same 40 assays. Its representation readout is +0.29441 [+0.25042, +0.33551], its R−M is unresolved and slightly negative at −0.03024 to −0.03250 across seeds, and its increment over the matched baseline is unresolved at −0.00410 to −0.00978. Supplying a genuine functional class therefore does not produce a resolved representation increment in the one arm that consumes it, and its rank-MSE reduction is resolved below zero at all three seeds, from −0.01915 to −0.03047.

ProteinGLM's increment is larger on this panel than on the anchor — +0.04542 to +0.05139, resolved above zero at all three seeds — while ProGen3-3B's becomes unresolved at +0.00307 to +0.01799 and the three Llama-lineage arms sit at or below zero. With 30 resampling units against 163 on the anchor, these intervals are about twice as wide, and the panel is far too small to rank arms; the one statement it supports is that function-conditioned readout is not where the representation increment becomes general.

## Native-support sensitivities

Fifteen arms have a length-eligible native support larger than the anchor and are refitted there at the primary seed only. These are within-arm sensitivities on the same biological cohort, not model comparisons: their supports differ from one another and from the anchor.

| Model | Stratum | Assays / clusters / variants | Raw M | Δρ R−M | Δρ (B+R)−B | MSE(B)−MSE(B+R) |
| --- | --- | --- | --- | --- | --- | --- |
| DialoGPT-small | Literal text AA | 211 / 169 / 26943 | -0.00833 [-0.03072, +0.01305] | +0.08208 [+0.05211, +0.11149] | -0.02307 [-0.03183, -0.01489] | +0.01357 [-0.02510, +0.08085] |
| GPT-2 | Literal text AA | 211 / 169 / 26943 | -0.00447 [-0.02301, +0.01503] | +0.12623 [+0.09502, +0.15653] | -0.02283 [-0.03390, -0.01336] | +0.01687 [-0.02320, +0.08599] |
| GPT-2-XL | Literal text AA | 211 / 169 / 26943 | -0.00681 [-0.02920, +0.01442] | +0.12077 [+0.08745, +0.15579] | -0.02106 [-0.03282, -0.00947] | +0.01874 [-0.02055, +0.08621] |
| GPT-2-large | Literal text AA | 211 / 169 / 26943 | -0.01378 [-0.03381, +0.00564] | +0.13339 [+0.09927, +0.16700] | -0.02770 [-0.03971, -0.01790] | +0.01560 [-0.02685, +0.08929] |
| GPT-2-medium | Literal text AA | 211 / 169 / 26943 | -0.00263 [-0.02225, +0.01545] | +0.09800 [+0.06606, +0.13058] | -0.02725 [-0.03959, -0.01738] | +0.01322 [-0.02826, +0.08473] |
| Llama-3.2-3B | Literal text AA | 212 / 170 / 27071 | -0.00086 [-0.01975, +0.01911] | +0.09530 [+0.06692, +0.12400] | -0.02053 [-0.03053, -0.01136] | +0.02883 [-0.02321, +0.12519] |
| Qwen2.5-0.5B | Literal text AA | 212 / 170 / 27071 | +0.01821 [-0.00138, +0.03704] | +0.07818 [+0.04816, +0.10648] | -0.02150 [-0.03166, -0.01277] | +0.02923 [-0.02519, +0.12837] |
| Qwen2.5-0.5B-Instruct | Literal text AA | 212 / 170 / 27071 | +0.00445 [-0.01342, +0.02106] | +0.08896 [+0.05967, +0.11914] | -0.02048 [-0.02822, -0.01373] | +0.02677 [-0.02732, +0.12456] |
| Qwen2.5-32B | Literal text AA | 212 / 170 / 27071 | +0.03053 [+0.00694, +0.05226] | +0.12860 [+0.09454, +0.16534] | -0.01318 [-0.02298, -0.00340] | +0.03511 [-0.01736, +0.13231] |
| Qwen2.5-7B | Literal text AA | 212 / 170 / 27071 | +0.00556 [-0.01242, +0.02397] | +0.12734 [+0.09370, +0.16114] | -0.01576 [-0.02699, -0.00520] | +0.03471 [-0.01804, +0.13187] |
| Qwen3-8B-Base | Literal text AA | 212 / 170 / 27071 | -0.00372 [-0.02381, +0.01666] | +0.09781 [+0.06978, +0.12707] | -0.02722 [-0.03798, -0.01782] | +0.02872 [-0.02668, +0.13213] |
| Llama-2-7B parent | Native sequence | 211 / 169 / 26943 | -0.00740 [-0.02713, +0.01226] | +0.14150 [+0.10902, +0.17449] | -0.01434 [-0.02469, -0.00517] | +0.02194 [-0.02022, +0.09480] |
| ProLLaMA Stage 1 | Native sequence | 211 / 169 / 26943 | +0.14604 [+0.11814, +0.17435] | +0.07758 [+0.04120, +0.11171] | -0.00755 [-0.02245, +0.00615] | +0.03037 [-0.01402, +0.10500] |
| ProLLaMA Stage 2 | Native sequence | 211 / 169 / 26943 | +0.15865 [+0.13170, +0.18655] | +0.08355 [+0.05023, +0.11371] | -0.00361 [-0.01759, +0.00946] | +0.03275 [-0.01252, +0.10782] |
| ProtGPT2 | Native sequence | 214 / 172 / 27327 | +0.33622 [+0.30712, +0.36355] | -0.05967 [-0.09141, -0.02698] | +0.00369 [-0.00630, +0.01386] | +0.04289 [-0.00351, +0.12579] |

Every native-support result preserves the sign pattern of its anchor result. The eleven literal-text arms stay resolved negative on the primary increment, from −0.01318 to −0.02770. The Llama-2-7B parent stays resolved negative at −0.01434 [−0.02469, −0.00517]; both ProLLaMA stages and ProtGPT2 stay unresolved. No native sensitivity moves an arm across zero.

## Fixed label-shuffle diagnostics

One fixed within-assay label permutation at seed 20269923 is used for coefficient fitting and inner tuning, then evaluated against the untouched held-out effects. The primary seed only is reported; the repeats do not rerun this control.

| Model | Panel | Shuffled-label B | Shuffled-label B+R |
| --- | --- | --- | --- |
| ByGPT5-base-en | anchor201 | -0.02497 [-0.03906, -0.01219] | -0.02647 [-0.04550, -0.00835] |
| ByGPT5-medium-en | anchor201 | -0.02391 [-0.03821, -0.01094] | -0.03520 [-0.05624, -0.01515] |
| ByGPT5-small-en | anchor201 | -0.02544 [-0.03930, -0.01248] | -0.04077 [-0.05951, -0.02287] |
| DialoGPT-small | anchor201 | -0.02444 [-0.03847, -0.01167] | -0.01999 [-0.03805, +0.00027] |
| GPT-2 | anchor201 | -0.02463 [-0.03865, -0.01183] | -0.01242 [-0.03184, +0.00560] |
| GPT-2-XL | anchor201 | -0.02480 [-0.03880, -0.01204] | -0.01978 [-0.03754, -0.00196] |
| GPT-2-large | anchor201 | -0.02452 [-0.03874, -0.01148] | -0.01936 [-0.03646, -0.00182] |
| GPT-2-medium | anchor201 | -0.02503 [-0.03922, -0.01191] | -0.02360 [-0.03979, -0.00711] |
| Galactica-1.3B | anchor201 | -0.02365 [-0.03794, -0.01073] | -0.05380 [-0.07457, -0.03153] |
| Galactica-125M | anchor201 | -0.02487 [-0.03908, -0.01194] | -0.03484 [-0.05390, -0.01489] |
| Galactica-30B | anchor201 | -0.02319 [-0.03759, -0.01018] | -0.06542 [-0.08796, -0.04354] |
| Galactica-6.7B | anchor201 | -0.02277 [-0.03695, -0.00974] | -0.06854 [-0.08721, -0.04856] |
| InstructProtein | anchor201 | -0.04135 [-0.05544, -0.02828] | -0.12436 [-0.15081, -0.09672] |
| Llama-2-7B parent | anchor201 | -0.02453 [-0.03877, -0.01142] | -0.02114 [-0.03835, -0.00396] |
| Llama-3.2-3B | anchor201 | -0.02554 [-0.03970, -0.01247] | -0.01941 [-0.03765, -0.00041] |
| ProGen2-base | anchor201 | -0.02925 [-0.04361, -0.01585] | -0.15144 [-0.18082, -0.12209] |
| ProGen2-large | anchor201 | -0.02563 [-0.04072, -0.01215] | -0.16538 [-0.19640, -0.13533] |
| ProGen2-medium | anchor201 | -0.04600 [-0.06063, -0.03243] | -0.14500 [-0.17484, -0.11575] |
| ProGen2-small | anchor201 | -0.04647 [-0.06073, -0.03297] | -0.06897 [-0.09824, -0.04060] |
| ProGen2-xlarge | anchor201 | -0.05359 [-0.06803, -0.04008] | -0.16728 [-0.19536, -0.13920] |
| ProGen3-112M | anchor201 | -0.03877 [-0.05327, -0.02554] | -0.06310 [-0.08756, -0.03917] |
| ProGen3-3B | anchor201 | -0.02725 [-0.04206, -0.01363] | -0.06388 [-0.09835, -0.03046] |
| ProLLaMA Stage 1 | anchor201 | -0.03168 [-0.04592, -0.01877] | -0.10440 [-0.12555, -0.08300] |
| ProLLaMA Stage 2 | anchor201 | -0.03730 [-0.05153, -0.02438] | -0.10484 [-0.12527, -0.08476] |
| ProtGPT2 | anchor201 | -0.06810 [-0.08271, -0.05369] | -0.15765 [-0.18047, -0.13319] |
| ProtGPT3-1.3B | anchor201 | -0.05714 [-0.07125, -0.04394] | -0.12409 [-0.14899, -0.09751] |
| ProteinGLM-7B-CLM | anchor201 | -0.03676 [-0.05153, -0.02342] | -0.08221 [-0.11547, -0.04809] |
| Qwen2.5-0.5B | anchor201 | -0.02385 [-0.03795, -0.01099] | -0.00868 [-0.02780, +0.00928] |
| Qwen2.5-0.5B-Instruct | anchor201 | -0.02450 [-0.03861, -0.01179] | -0.02089 [-0.03857, -0.00404] |
| Qwen2.5-32B | anchor201 | -0.02159 [-0.03586, -0.00861] | -0.05019 [-0.06892, -0.03255] |
| Qwen2.5-7B | anchor201 | -0.02305 [-0.03737, -0.01005] | -0.03139 [-0.04860, -0.01329] |
| Qwen3-8B-Base | anchor201 | -0.02521 [-0.03940, -0.01226] | -0.01560 [-0.03586, +0.00402] |
| RITA-XL | anchor201 | -0.04151 [-0.05588, -0.02833] | -0.18630 [-0.21214, -0.15950] |
| Llama-2-7B parent | EC_conditioned40 | -0.00231 [-0.03485, +0.02688] | -0.02566 [-0.05056, -0.00260] |
| ProGen3-3B | EC_conditioned40 | +0.01691 [-0.01591, +0.04642] | -0.08886 [-0.15281, -0.03093] |
| ProLLaMA Stage 1 | EC_conditioned40 | -0.00383 [-0.03519, +0.02448] | -0.02263 [-0.07138, +0.02678] |
| ProLLaMA Stage 2 | EC_conditioned40 | -0.00957 [-0.04170, +0.01841] | -0.05275 [-0.09973, -0.00589] |
| ProteinGLM-7B-CLM | EC_conditioned40 | +0.00495 [-0.02762, +0.03384] | +0.07255 [-0.00679, +0.14648] |
| ZymCTRL | EC_conditioned40 | +0.00421 [-0.02757, +0.03222] | -0.15215 [-0.20194, -0.09968] |
| DialoGPT-small | native_dialogpt-small | +0.01136 [-0.00458, +0.02675] | +0.01325 [-0.00706, +0.03216] |
| GPT-2 | native_gpt2 | +0.01159 [-0.00411, +0.02703] | +0.02380 [+0.00601, +0.04332] |
| GPT-2-large | native_gpt2-large | +0.01054 [-0.00502, +0.02581] | +0.01741 [-0.00342, +0.03784] |
| GPT-2-medium | native_gpt2-medium | +0.01136 [-0.00444, +0.02657] | +0.01129 [-0.00884, +0.03179] |
| GPT-2-XL | native_gpt2-xl | +0.01097 [-0.00482, +0.02618] | +0.00003 [-0.01816, +0.01814] |
| Llama-2-7B parent | native_llama-2-7b | +0.01174 [-0.00383, +0.02705] | -0.00225 [-0.02017, +0.01536] |
| Llama-3.2-3B | native_llama-3.2-3b | +0.01243 [-0.00158, +0.02735] | +0.00737 [-0.00935, +0.02568] |
| ProLLaMA Stage 2 | native_prollama | +0.01157 [-0.00450, +0.02692] | +0.04254 [+0.02063, +0.06417] |
| ProLLaMA Stage 1 | native_prollama-stage-1 | +0.01877 [+0.00280, +0.03436] | -0.01968 [-0.04220, +0.00170] |
| ProtGPT2 | native_protgpt2 | -0.00515 [-0.02036, +0.01030] | -0.05390 [-0.07821, -0.02991] |
| Qwen2.5-0.5B | native_qwen2.5-0.5b | +0.01303 [-0.00112, +0.02767] | +0.00024 [-0.01713, +0.01870] |
| Qwen2.5-0.5B-Instruct | native_qwen2.5-0.5b-instruct | +0.01234 [-0.00181, +0.02697] | +0.00587 [-0.00986, +0.02294] |
| Qwen2.5-32B | native_qwen2.5-32b | +0.01375 [-0.00056, +0.02868] | -0.02594 [-0.04424, -0.00656] |
| Qwen2.5-7B | native_qwen2.5-7b | +0.01335 [-0.00071, +0.02795] | -0.00264 [-0.02149, +0.01645] |
| Qwen3-8B-Base | native_qwen3-8b-base | +0.01216 [-0.00207, +0.02710] | +0.00461 [-0.01241, +0.02243] |

On the anchor panel, all 33 shuffled-label B values are resolved below zero, and 29 of 33 shuffled-label B+R values are resolved below zero, with DialoGPT-small, GPT-2, Qwen2.5-0.5B and Qwen3-8B-Base unresolved. On the 15 native panels the picture changes sign: ProLLaMA Stage 1's shuffled-label B is resolved *above* zero at +0.01877 [+0.00280, +0.03436], and so are GPT-2's and ProLLaMA Stage 2's shuffled-label B+R at +0.02380 and +0.04254, while ProtGPT2's and Qwen2.5-32B's shuffled-label B+R are resolved below zero. On the EC panel all six shuffled-label B values are unresolved and four of six shuffled-label B+R values are resolved below zero, ZymCTRL's most strongly at −0.15215 [−0.20194, −0.09968].

These outcomes must not be summarized as uniformly zero or as a universally passed control. A single fixed permutation, assessed with intervals conditional on its own fitted predictions, is not a calibrated randomization test: the intervals do not propagate variation across permutations. The results bound how strongly this diagnostic can exclude fitting artifacts; they neither supply permutation p-values nor by themselves demonstrate label leakage. No post-hoc permutation experiment was added.

## Remaining equally supervised controls

These predictors use the same folds, labels, weighting and ridge tuning. Sequence-only uses the 444 sequence descriptors; profile-plus-sequence adds P without M or R. The values below are the original five arms at the primary seed; on the anchor panel fitted P is +0.36066 [+0.33207, +0.38910], sequence-only is +0.30024 [+0.27330, +0.32841] and profile-plus-sequence is +0.43638 [+0.40676, +0.46567] for every arm, by the verified exact agreement of those predictions. The added arms' arm-specific fitted M and P+M values remain in their admitted reports.

| Model | Fitted P | Fitted M | P+M | Sequence only | Profile + sequence |
| --- | --- | --- | --- | --- | --- |
| ProGen3-3B | +0.36066 [+0.33207, +0.38910] | +0.37866 [+0.35136, +0.40529] | +0.41727 [+0.38954, +0.44326] | +0.30024 [+0.27330, +0.32841] | +0.43638 [+0.40676, +0.46567] |
| ProteinGLM | +0.36066 [+0.33207, +0.38910] | +0.38984 [+0.36161, +0.41669] | +0.42144 [+0.39413, +0.44775] | +0.30024 [+0.27330, +0.32841] | +0.43638 [+0.40676, +0.46567] |
| Llama-2 parent | +0.36066 [+0.33207, +0.38910] | -0.01037 [-0.02902, +0.00926] | +0.36372 [+0.33544, +0.39254] | +0.30024 [+0.27330, +0.32841] | +0.43638 [+0.40676, +0.46567] |
| ProLLaMA Stage 1 | +0.36066 [+0.33207, +0.38910] | +0.14338 [+0.11456, +0.17101] | +0.36083 [+0.33240, +0.38897] | +0.30024 [+0.27330, +0.32841] | +0.43638 [+0.40676, +0.46567] |
| ProLLaMA Stage 2 | +0.36066 [+0.33207, +0.38910] | +0.15595 [+0.12674, +0.18482] | +0.36122 [+0.33283, +0.38868] | +0.30024 [+0.27330, +0.32841] | +0.43638 [+0.40676, +0.46567] |

The large B−M gains in the tables above — up to +0.45639 for ByGPT5-medium-en, whose likelihood ranks nothing — show that most of the improvement available over a native likelihood readout is already obtained by supervised profile and sequence controls, in every stratum.

## Readout problem or representation ceiling

The expansion was run to separate a readout problem, in which the information is present in the representation but the fitted class cannot extract it, from a representation ceiling, in which the generalisable representation stops at a lower level of biological organisation. The 34-arm panel does not support the readout explanation as a panel-wide account, and it does not establish the ceiling either. What it establishes is narrower and specific to the rung being tested.

Three findings carry that verdict. First, the likelihood readout is measurably suboptimal: R−M is resolved above zero in 69 of the 99 anchor arm-seed cells, with the largest gains in exactly the arms whose likelihood ranks worst. Second, that suboptimality is not additional information: once M, P and the sequence descriptors are fitted together on the same labels and folds, adding 1,024 representation coordinates does not raise the held-cluster Spearman with a resolved interval in 30 of the 33 anchor arms, or in 93 of the 99 anchor arm-seed cells. Third, the exceptions are not arbitrary. The only arm resolved above zero at all three seeds is ProteinGLM-7B-CLM, at +0.01970 to +0.02255; the only other resolved arms are ProGen3-3B and ProGen2-base, both native protein interfaces. No text arm on literal strings, no Galactica size, no ProLLaMA stage and no conditioned arm reaches a resolved positive increment at any seed.

For the readout rung the verdict is therefore: a readout gain beyond the native likelihood exists and is widespread, and a readout gain beyond the matched supervised baseline does not generalise across the panel. Within this compressed linear class, 31 of 34 arms place no resolved additional mutation-effect information in their frozen middle-block and final-block pooled states beyond what profile and sequence descriptors already carry.

What that licenses for the capability hierarchy is bounded in a way worth stating plainly. A representation increment that survives profile controls is evidence about *this readout of this representation*, not about the representation as such and not about the readout class as such. A resolved positive increment shows that a linear function of the frozen states carries mutation-effect information that the matched supervised controls do not; it does not show where that information came from, that the model computes it as a biological mechanism, or that it would survive an intervention. A null or negative increment shows that this readout does not reach such information; it does not show the information is absent, because linearity, four pooled summaries, a fixed 1,024-coordinate Gaussian projection, a finite phenotype-label budget of 163 clusters and transfer to unseen clusters all bound the measurement independently. The panel cannot distinguish "no information" from "no linearly accessible information at these two depths under this compression".

Separating the two explanations further requires changing one factor at a time on the same frozen states. Varying the readout class while holding the representation fixed — nonlinear heads, a per-layer sweep instead of two depths, position-resolved or attention-pooled features, and the uncompressed features that the retained full-dimensional vectors already permit to be replayed without model inference — would test the readout explanation directly, since the current class is the binding constraint on any negative result. Varying the label budget would test whether the nulls are power-bound at 163 clusters. Establishing a claim about the representation rather than about a readout requires an intervention: removing the information from the states and measuring the behavioural change, which converts a correlational readout statement into a causal one. None of that is authorized by this execution, which ends with this report.

## Associations across the panel

These are descriptive groupings of unmatched checkpoints. They cannot identify causal effects of architecture, modality, objective, scale or training corpus, and the counts of resolved intervals are neither corrected significance nor model-population prevalence.

The increment does not scale with model size. Inside the ProGen2 ladder, which holds tokenizer and architecture fixed across five sizes, the primary-seed increment runs +0.01243, +0.00981, +0.01341, +0.00906 and +0.00075 for small, medium, base, large and xlarge, so the largest checkpoint — the one with the highest raw-likelihood point estimate in the panel at +0.39206 — has the smallest increment. Inside the four Galactica sizes the increment is negative in point estimate at every size, from −0.01225 to −0.00769, while raw M rises steeply with size from +0.00328 to +0.13590. Inside the GPT-2 ladder the four increments span −0.02083 to −0.02894 and inside ByGPT5 the three span −0.01659 to −0.02127, in both cases with no ordering by size; inside Qwen2.5 the resolved-negative increment shrinks in magnitude from −0.02060 at 0.5B to −0.01115 at 32B without crossing zero. The single within-lineage case where the increment rises with size is ProGen3, from an unresolved +0.00666 at 112M to a resolved +0.01736 at 3B.

The increment does track the pretraining corpus and the interface, in the weak sense that every resolved positive increment in the panel belongs to a native protein-sequence interface trained on protein sequence. It does not follow that a native protein corpus is sufficient: 16 of the 19 native-sequence arms produce no resolved positive increment at any seed, among them all four Galactica sizes, RITA-XL, ProtGPT2, ProtGPT3-1.3B, InstructProtein, ProGen3-112M and four of the five ProGen2 sizes. Within the one lineage that isolates protein continued pretraining on a fixed parent, the Llama-2-7B parent's increment is resolved negative at all three seeds while both ProLLaMA stages are unresolved at the primary seed, so protein adaptation moves the increment toward zero from below without establishing a positive one. Neither Qwen2.5-0.5B against Qwen2.5-0.5B-Instruct nor ProGen2-medium against ProGen2-base, the two pairs that vary instruction tuning and pretraining mixture at fixed size, separates by more than the interval width.

## The largest capability difference in the panel

The behavioural anchor for any later mechanism study is the native likelihood's mutation-effect ranking, not the representation increment. On identical support — 201 assays, 163 clusters, 25,728 variants, primary seed — raw M spans ProGen2-xlarge at +0.39206 [+0.36438, +0.41839] and ProteinGLM-7B-CLM at +0.38984 [+0.36161, +0.41669] down to ByGPT5-medium-en at −0.02027 [−0.04147, +0.00086], with 13 of the 14 literal-text arms unresolved against zero. That is the panel's largest and cleanest capability difference, and it is a difference in likelihood assignment rather than in what a fitted readout can recover: the same 14 text arms whose likelihoods carry no resolved ranking signal support representation readouts of +0.06753 to +0.15596.

The sharpest matched version of that difference is the ProLLaMA lineage, where protein continued pretraining acts on one fixed parent checkpoint. On the 211-assay native support the parent's raw M is −0.00740 [−0.02713, +0.01226], Stage 1's is +0.14604 [+0.11814, +0.17435] and Stage 2's is +0.15865 [+0.13170, +0.18655]. The gap between the parent and either stage is larger than either interval's width, though the difference itself was not fitted as a paired contrast and therefore carries no interval of its own. A mechanism study of how protein likelihood assignment arises has a well-posed target here: two stages and their parent, same architecture, same tokenizer, same cohort, with the capability appearing between them.

## Limitations

Five re-extracted arms carry a structurally vacuous requalification. ProtGPT3-1.3B, Galactica-125M, DialoGPT-small, Qwen2.5-7B and ByGPT5-medium-en exceeded a fixed 0.001 production gate at batch size eight and were re-run whole-arm at batch size one. At batch size one the production scoring forward and the repeat check's reference forward in `scripts/transfer/extract_frozen_readout.py` are the same single-row computation over the same string, so their exactly-zero drift is true by construction and certifies nothing about numerical accuracy. Their extractions are admitted and internally consistent and must not be described as numerically validated. The same vacuity applies to every batch-one arm in the panel, 20 of 34:

| Model | Stratum | Precision | Hidden width | Assays |
| --- | --- | --- | ---: | ---: |
| ByGPT5-medium-en | literal_text_AA | float32 | 1536 | 201 |
| DialoGPT-small | literal_text_AA | float32 | 768 | 211 |
| Galactica-125M | native_sequence | float32 | 768 | 201 |
| Galactica-30B | native_sequence | float32 | 7168 | 201 |
| Llama-2-7B parent | native_sequence | float32 | 4096 | 211 |
| Llama-3.2-3B | literal_text_AA | float32 | 3072 | 212 |
| ProGen2-base | native_sequence | float32 | 1536 | 201 |
| ProGen2-large | native_sequence | float32 | 2560 | 201 |
| ProGen2-medium | native_sequence | float32 | 1536 | 201 |
| ProGen2-small | native_sequence | float32 | 1024 | 201 |
| ProGen2-xlarge | native_sequence | float32 | 4096 | 201 |
| ProGen3-112M | native_sequence | bfloat16 | 384 | 201 |
| ProGen3-3B | native_sequence | bfloat16 | 1280 | 201 |
| ProLLaMA Stage 1 | native_sequence | float32 | 4096 | 211 |
| ProLLaMA Stage 2 | native_sequence | float32 | 4096 | 211 |
| ProtGPT3-1.3B | native_sequence | float32 | 1024 | 201 |
| Qwen2.5-0.5B-Instruct | literal_text_AA | float32 | 896 | 212 |
| Qwen2.5-32B | literal_text_AA | float32 | 5120 | 212 |
| Qwen2.5-7B | literal_text_AA | float32 | 3584 | 212 |
| Qwen3-8B-Base | literal_text_AA | float32 | 4096 | 212 |

The batch-eight exits that forced those five re-extractions were real measurements of batch-composition sensitivity and are worth reporting as such. ProtGPT3-1.3B exceeded both gates on `NRAM_I33A0_Jiang_2016` after 96 checked assays, at 0.015167236328125 nats and 0.005362214307723598 mutation-vector relative L2. Galactica-125M reached 0.00152587890625 nats on `GFP_AEQVI_Sarkisyan_2016` after 61 assays; DialoGPT-small reached 0.001953125 nats on `A0A192B1T2_9HIV1_Haddox_2018` on its first assay; Qwen2.5-7B reached 0.0010776871634436153 relative L2 on that same assay on its first; ByGPT5-medium-en reached 0.001183065839635574 relative L2 on `TPOR_HUMAN_Bridgford_2020` after 188 assays. These are finite single-assay maxima with no sampling interval.

The 14 arms that ran at batch size eight do carry informative repeat checks, and all of their observed production maxima sit below both gates, with Qwen2.5-0.5B the closest at 0.00099892176436380776 relative L2 against a limit of 0.001:

| Model | Hidden width | Assays checked | Max likelihood-delta drift (nats) | Max mutation-vector relative L2 |
| --- | ---: | ---: | --- | --- |
| ByGPT5-base-en | 1536 | 201 | 0.000732421875 | 0.00079611740642883248 |
| ByGPT5-small-en | 1472 | 201 | 0.00048828125 | 0.00080792294326523305 |
| GPT-2 | 768 | 211 | 0.0009765625 | 0.00082031909701275212 |
| GPT-2-XL | 1600 | 211 | 0.00048828125 | 0.00065472974409557828 |
| GPT-2-large | 1280 | 211 | 0.00048828125 | 0.00056091755285030229 |
| GPT-2-medium | 1024 | 211 | 0.000732421875 | 0.00048779246814609483 |
| Galactica-1.3B | 2048 | 201 | 0.000335693359375 | 0.00045160137970573798 |
| Galactica-6.7B | 4096 | 201 | 0.00054931640625 | 0.00012659296518748524 |
| InstructProtein | 2048 | 201 | 0.0003662109375 | 0.00015636270051878835 |
| ProtGPT2 | 1280 | 214 | 0.000732421875 | 0.00097185095223525518 |
| ProteinGLM-7B-CLM | 4096 | 201 | 0.00023651123046875 | 9.7202250093920978e-05 |
| Qwen2.5-0.5B | 896 | 212 | 0.000732421875 | 0.00099892176436380776 |
| RITA-XL | 2048 | 201 | 0.00028228759765625 | 0.00070936823612976715 |
| ZymCTRL | 1280 | 40 | 0.000244140625 | 0.00018643309376759399 |

ProGen3 checkpoints are singleton-only, and this bounds any ProGen3 number produced under batching. `sequence_ids` indexes a learned `nn.Embedding(512, hidden_size)` whose value must be the native independent-sequence zero, and `SparseMoeBlock.forward` flattens batch and position into one token axis and runs one gather-and-GEMM per expert over the whole flattened batch, with no kernel above float16 in the release. With the sequence-identifier assignment corrected, batched-minus-singleton target negative log likelihood still reaches 0.02464 nats per scored token for ProGen3-3B at batch size two (mean 0.00658) and 0.02796 for ProGen3-112M at batch size eight (mean 0.01252); eight identical copies of one 770-token row agree exactly with each other while sitting 0.02444 nats per token from the singleton value, and sixteen equal-length rows disagree by up to 0.07179, so the dependence is on batch extent rather than on batch position or padding. The scoring path now refuses any batch size above one on this packing. Both ProGen3 arms in this panel ran at batch size one in bfloat16 and are within that bound; no ProGen3 quantity produced under batching is admissible. The historical Stage-46 retractions remain in force.

The remaining bounds are properties of the design rather than of its execution. The readout class is one fixed compressed linear family at two depths, so no negative result here reaches other layers, nonlinear readouts, uncompressed features or richer profile baselines. The label budget is 163 clusters on the anchor and 30 on the EC panel, and the EC panel's intervals are correspondingly about twice as wide. Wild-type clustering at 50% identity does not guarantee remote-superfamily disjointness or exclude pretraining overlap. The three split seeds are robustness checks on one biological cohort, not independent replications, and the minimum-to-maximum point ranges reported with them are not uncertainty intervals. Checkpoint lineages inside the panel are correlated, so counts of resolved intervals do not estimate a model-population prevalence. The two ProtGPT2 rows retain native formatted-target likelihood, which may include predicted formatting tokens, rather than residue-only likelihood. ZymCTRL's conditioned results cannot attribute prediction to sequence weights or to the supplied annotation.

## Scope of the conclusion

The supported distinction is between improved readout relative to the native likelihood, which this panel demonstrates broadly, and additional prediction beyond equally supervised controls, which it demonstrates in 3 of 34 arms at magnitudes up to +0.02255 Spearman and in one arm at all three split seeds. ProteinGLM-7B-CLM supplies the only split-consistent positive increment; ProGen3-3B supplies a smaller split-sensitive one; ProGen2-base supplies one resolved at a single seed. Every literal-text arm is resolved in the opposite direction. A panel-wide near-null on the primary endpoint is the result, and it is reported as such rather than resolved by selecting a split, a stratum or a subset of arms.

That result does not justify a blanket statement that readout mismatch is absent, nor that it explains native model deficits, nor that any model's representation contains or lacks a particular biological computation. It bounds one rung: within a fixed compressed linear readout of two pooled depths, at this label budget, on this cohort, mutation-effect information beyond profile and sequence descriptors is not generally accessible in frozen representations, and where it is accessible the effect is small and confined to native protein-sequence interfaces. This authorization ends with this report; no hierarchy, objective, missing-condition or epistasis follow-up is part of it.
