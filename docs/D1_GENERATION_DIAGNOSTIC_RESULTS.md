# Generation failure diagnostic results

## Scope and admission

The four residue-budget confirmation cells and four historical-ledger decompositions completed on H200 on 2026-09-23 under the [frozen diagnostic protocol](D1_GENERATION_DIAGNOSTIC_PROTOCOL.md). The confirmation asks which observable stages accompany low profile recognition at a fixed native unlabelled operating point. It does not establish biological completeness, folding or function. Component swaps are a separate experiment and are not included here.

The admitted transfer archive has SHA256 `e8db446d423f1bbc503603a28f12fdca56f95b15f464afaf3b56e8a911dff55c`. Local evidence is under `results/transfer/d1_followup_20260923/generation/`: run `20260923062631_dfd71cd7e171` contains the four `genfail_*` cells and run `20260923063207_0c10dded9867` the four `genhistory_*` cells. Each cell retains its attempt ledger, summary and summary checksum. Confirmation cells also retain token traces and Pfam search tables. All eight checksums, ledger lengths, raw-output classifications, outcome counts, stop counts and deterministic shingle-group summaries were checked against the saved summaries. Local verification performed no model inference or resampling.

Percentages below use all attempts as denominator and report saved pointwise Wilson 95% intervals. These intervals concern stochastic attempts under the specified generation procedure, not model-training uncertainty, biological validation or family generalization; they are not adjusted for multiple comparisons. Group counts and stop-reason counts are exact descriptive census counts without an estimated interval unless explicitly provided.

## Residue-budget confirmation

Each cell contains 64 attempts with a 400-residue stopping budget and a separate 1024-token safety cap. Sampling follows the protocol; all four checkpoints use bfloat16 here. Native-format completion requires a nonempty canonical sequence followed by the checkpoint's terminal. A Pfam-positive censored prefix remains prefix evidence.

| Checkpoint | Native complete | Complete without Pfam | Complete with Pfam | Any Pfam |
| --- | --- | --- | --- | --- |
| ProLLaMA Stage 1 | 41/64: 64.1% [51.8, 74.7] | 34/64: 53.1% [41.1, 64.8] | 7/64: 10.9% [5.4, 20.9] | 9/64: 14.1% [7.6, 24.6] |
| ProLLaMA Stage 2 | 59/64: 92.2% [83.0, 96.6] | 54/64: 84.4% [73.6, 91.3] | 5/64: 7.8% [3.4, 17.0] | 6/64: 9.4% [4.4, 19.0] |
| ProGen3-112M | 41/64: 64.1% [51.8, 74.7] | 26/64: 40.6% [29.5, 52.9] | 15/64: 23.4% [14.7, 35.1] | 22/64: 34.4% [23.9, 46.6] |
| ProGen3-3B | 38/64: 59.4% [47.1, 70.5] | 13/64: 20.3% [12.3, 31.7] | 25/64: 39.1% [28.1, 51.3] | 34/64: 53.1% [41.1, 64.8] |

No checkpoint produced an empty raw output, empty canonical prefix or format failure: each event was 0/64 attempts, 0.0% [0.0, 5.7] per checkpoint. Every stop was either a native terminal or the residue budget; no safety-cap or standalone EOS stop occurred (0/64 per cell, descriptive counts). Native-terminal counts equal native-complete counts in the table. ProGen3's terminal contains EOS, whereas the ProLLaMA runner stops at `>` before EOS; EOS presence is therefore an interface-specific field, not a cross-model quality measure.

| Checkpoint | Residue-budget stops / attempts, percent [95% interval] | Pfam-bearing groups / all groups | Complete Pfam-bearing groups / all groups | Residue-cap overshoot |
| --- | --- | --- | --- | --- |
| ProLLaMA Stage 1 | 23/64, 35.9% [25.3, 48.2] | 9/64 | 7/64 | Four attempts by 1 residue; one by 2 residues |
| ProLLaMA Stage 2 | 5/64, 7.8% [3.4, 17.0] | 6/64 | 5/64 | One attempt by 1 residue |
| ProGen3-112M | 23/64, 35.9% [25.3, 48.2] | 22/64 | 15/64 | None |
| ProGen3-3B | 26/64, 40.6% [29.5, 52.9] | 34/64 | 25/64 | None |

The budget intervals are the complements of the saved native-completion Wilson intervals: those two outcomes partition every new cell. All other attempts had zero overshoot. Grouping uses connected components of residue 5-shingle containment at threshold 0.5; all 64 attempts in each cell are separate groups, so deduplication removes no recognized product in these draws. Groups are sequence groups, not distinct Pfam families, and no group-count confidence interval is estimated.

Stage 2's low recognition occurs predominantly among format-valid, normally terminated products: 54/64 attempts are complete without Pfam, as quantified above. Its confirmation is therefore not dominated by empty strings, formatting failures, censoring or between-output near-duplication. The Stage 1 and Stage 2 recognition intervals are broad; this small confirmation alone does not establish their rate difference or equivalence. The complete-product descriptor census below characterizes length, composition and within-sequence repetition without isolating their causal contributions.

ProGen3 recognition also needs a completeness qualification. Its all-attempt Pfam counts exceed its complete-with-Pfam counts in both cells; the gap consists of recognized residue-budget prefixes. A recognition-only endpoint would hide that distinction. The data do not establish whether extending those prefixes would yield complete profile-bearing proteins.

## Historical attempt decomposition

Each historical cell contains 800 previously generated attempts. Pfam flags are inherited from the annotated source ledgers; no generation was repeated for this decomposition. The historical ledgers lack exact token stopping traces, so an absent native terminal is classified as unknown stopping, not proven truncation.

| Checkpoint | Native complete | Complete without Pfam | Complete with Pfam | Any Pfam |
| --- | --- | --- | --- | --- |
| ProLLaMA Stage 1 | 579/800: 72.4% [69.2, 75.4] | 494/800: 61.8% [58.3, 65.1] | 85/800: 10.6% [8.7, 13.0] | 136/800: 17.0% [14.6, 19.8] |
| ProLLaMA Stage 2 | 743/800: 92.9% [90.9, 94.5] | 709/800: 88.6% [86.2, 90.6] | 34/800: 4.2% [3.1, 5.9] | 36/800: 4.5% [3.3, 6.2] |
| ProGen3-112M | 413/800: 51.6% [48.2, 55.1] | 257/800: 32.1% [29.0, 35.4] | 156/800: 19.5% [16.9, 22.4] | 252/800: 31.5% [28.4, 34.8] |
| ProGen3-3B | 501/800: 62.6% [59.2, 65.9] | 161/800: 20.1% [17.5, 23.0] | 340/800: 42.5% [39.1, 46.0] | 496/800: 62.0% [58.6, 65.3] |

| Checkpoint | Unknown stop / 800 attempts | Format failures / 800 attempts, percent [95% interval] | Pfam-bearing groups / all groups | Complete Pfam-bearing groups / all groups |
| --- | --- | --- | --- | --- |
| ProLLaMA Stage 1 | 221 | 0/800, 0.0% [0.0, 0.5] | 136/800 | 85/800 |
| ProLLaMA Stage 2 | 57 | 0/800, 0.0% [0.0, 0.5] | 36/786 | 34/786 |
| ProGen3-112M | 387 | 1/800, 0.1% [0.0, 0.7] | 252/800 | 156/800 |
| ProGen3-3B | 299 | 0/800, 0.0% [0.0, 0.5] | 496/800 | 340/800 |

Empty raw outputs and empty canonical prefixes each occurred in 0/800 attempts per checkpoint, 0.0% [0.0, 0.5]. Unknown-stop and group columns are exact descriptive counts with no inferential interval. Stage 2's historical near-duplicate grouping reduces the total group count but leaves every Pfam-positive attempt in a separate group. The historical Stage 2 recognition deficit therefore persists in the count of recognized groups and among completed products; empty outputs, invalid format and repeated recognized products do not account for it. This remains specific to the bare `Seq=<` operating point, outside the supervised instruction prompt.

## Length and repetition of completed products

The H200 descriptor census completed on 2026-09-23 from snapshot `20260923080747_a62315987c69`, without new generation. Its artifact, `results/transfer/d1_followup_20260923/generation_descriptors/generation_product_descriptors.json`, has SHA256 `bf866e6da7aca0c3bf76c38cc75a8baa3686e10dd057cc7b00ee9251759355ff`. All eight source hashes match the admitted diagnostic ledgers; complete-product identities, lengths, profile flags, group counts, metric availability and saved quantile bounds reconcile. The artifact retains every complete product’s continuous descriptors and full summaries separately for Pfam-positive and Pfam-negative products.

The following values are medians [25th, 75th percentiles] of the observed complete-product census, using linear interpolation; brackets are interquartile ranges, not confidence intervals. Entropy is the Shannon entropy of residue composition in nats per residue. Run fraction is the longest identical-residue run divided by sequence length. Unique 3-mer fraction is the number of distinct overlapping residue 3-mers divided by their number of positions. A sequence shorter than three residues has an undefined 3-mer fraction; none of these complete products falls in that category. No failure threshold is applied.

| Cohort / checkpoint | Complete products | Length, residues | Entropy, nats/residue | Run fraction | Unique 3-mer fraction |
| --- | ---: | --- | --- | --- | --- |
| Confirmation / ProLLaMA Stage 1 | 41 | 107.0 [89.0, 245.0] | 2.575 [2.410, 2.605] | 0.023 [0.015, 0.039] | 0.937 [0.903, 0.963] |
| Confirmation / ProLLaMA Stage 2 | 59 | 95.0 [76.5, 131.5] | 2.482 [2.342, 2.564] | 0.032 [0.023, 0.042] | 0.943 [0.914, 0.975] |
| Confirmation / ProGen3-112M | 41 | 220.0 [132.0, 298.0] | 2.584 [2.410, 2.680] | 0.017 [0.012, 0.022] | 0.942 [0.886, 0.966] |
| Confirmation / ProGen3-3B | 38 | 236.0 [155.5, 323.25] | 2.668 [2.590, 2.780] | 0.014 [0.009, 0.019] | 0.953 [0.919, 0.963] |
| Historical / ProLLaMA Stage 1 | 579 | 156.0 [100.0, 254.5] | 2.504 [2.323, 2.613] | 0.020 [0.013, 0.031] | 0.921 [0.820, 0.961] |
| Historical / ProLLaMA Stage 2 | 743 | 104.0 [72.0, 145.0] | 2.491 [2.337, 2.596] | 0.029 [0.022, 0.042] | 0.952 [0.877, 0.978] |
| Historical / ProGen3-112M | 413 | 221.0 [136.0, 291.0] | 2.591 [2.489, 2.684] | 0.015 [0.011, 0.022] | 0.940 [0.894, 0.966] |
| Historical / ProGen3-3B | 501 | 224.0 [143.0, 296.0] | 2.668 [2.568, 2.760] | 0.014 [0.010, 0.019] | 0.948 [0.912, 0.969] |

Length stratification by profile recognition retains the actual number of complete products in each group; the corresponding entropy and repetition distributions remain in the artifact.

| Cohort / checkpoint | Complete with Pfam: n; length median [Q25, Q75], residues | Complete without Pfam: n; length median [Q25, Q75], residues |
| --- | --- | --- |
| Confirmation / ProLLaMA Stage 1 | 7; 271.0 [252.0, 333.0] | 34; 100.5 [82.25, 139.5] |
| Confirmation / ProLLaMA Stage 2 | 5; 126.0 [102.0, 136.0] | 54; 94.5 [73.75, 129.75] |
| Confirmation / ProGen3-112M | 15; 289.0 [236.5, 345.0] | 26; 155.0 [119.75, 242.5] |
| Confirmation / ProGen3-3B | 25; 259.0 [226.0, 342.0] | 13; 151.0 [130.0, 215.0] |
| Historical / ProLLaMA Stage 1 | 85; 271.0 [208.0, 350.0] | 494; 142.0 [92.0, 219.0] |
| Historical / ProLLaMA Stage 2 | 34; 126.5 [93.75, 225.25] | 709; 103.0 [71.0, 144.0] |
| Historical / ProGen3-112M | 156; 262.0 [195.25, 309.25] | 257; 187.0 [119.0, 267.0] |
| Historical / ProGen3-3B | 340; 247.5 [171.75, 312.25] | 161; 170.0 [117.0, 240.0] |

Stage 2’s low recognition coexists with shorter terminated products in both cohorts. Its historical complete-output median is 104 residues compared with Stage 1’s 156 residues, and its confirmation median is 95 versus 107 residues, on the supports and interquartile ranges above. This supports a length-distribution difference at this operating point, but does not establish premature termination: a native terminal is a formatting event, and a short sequence can be biologically complete. Conditioning on completion and then Pfam recognition also selects different output populations across models.

The repetition descriptors do not support a uniform increase in Stage 2 repetition. In the historical complete-output census, Stage 2 has a higher longest-run fraction but also a higher unique 3-mer fraction than Stage 1; their composition-entropy medians are close, as quantified above. In the confirmation, entropy is lower for Stage 2, but the unique 3-mer medians again do not decrease. Run fractions increase mechanically when the same run occurs in a shorter sequence, and longer sequences provide more opportunities for repeated 3-mers. These are descriptive, length-dependent quantities; they do not isolate repetition or composition as the cause of low Pfam recognition. The length and within-sequence distributions complement the between-output deduplication census without establishing a causal mechanism.

## What the old/new comparison can establish

The historical [unconditional census](D1_UNCONDITIONAL_GENERATION_EXPANSION.md) and [ProGen3 campaign](D1_PROGEN3_GENERATION_PREREGISTRATION.md) specify the same temperature 0.85, top-p 0.95, top-k 50, repetition penalty 1 and batch size eight as the confirmation. Their seed is 20260905 plus batch index; the confirmation uses 20260923 plus batch index. The historical cap is 400 new tokens for ProLLaMA and an effective 402 tokens through ProGen3's official wrapper. The confirmation uses an explicit per-attempt native-terminal/residue stopping rule with token traces. These are independent draws with changed stopping implementation and sample size, not paired continuations or a budget-only randomized intervention. ProLLaMA tokenization can cross the residue boundary, as the retained overshoots show.

The two batches support descriptive consistency checks and a clearer decomposition of observed outcomes. They do not identify a causal token-budget effect by subtracting historical and new recognition percentages. The strongest current conclusion is narrower: low Stage 2 recognition at this native unlabelled operating point coexists with frequent valid termination, and cannot be assigned solely to a tokenizer-induced empty/formatting failure or to the removal of repeated recognized outputs. The descriptor census further associates this endpoint with shorter terminated products, while a causal account of that length difference or a parameter mechanism still requires a controlled analysis.
