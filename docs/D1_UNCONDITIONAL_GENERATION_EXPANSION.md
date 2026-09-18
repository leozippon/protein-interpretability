# Unconditional generation census expansion

Date: 2026-09-18 (Asia/Shanghai). Campaign: EXP-R2-246. This document freezes the missing native unlabelled generation points before those outputs exist. It does not change [EXP-R2-232](D1_GENERATION_BIOLOGY_PREREGISTRATION.md), [EXP-R2-233](D1_PROGEN3_GENERATION_PREREGISTRATION.md), or the completed ProGen3-3B ledger.

## Question

What does each admitted checkpoint produce at one fixed native unlabelled operating point, and how strong is the same bounded computational evidence already used for ProGen3-3B? The four endpoints are completion and any-Pfam recognition, generation-minus-own-shuffle CA-pLDDT, the all-attempt confidence-event estimate, and UniRef50 reference support. There is no requested class, so target-profile hits and requested-minus-mismatched contrasts are undefined.

This is not a common-task ranking against the ZymCTRL EC or ProLLaMA superfamily tasks. A budget-censored prefix is not a complete product. A Pfam hit is not function. A UniRef50 alignment is not novelty or a training-disjointness certificate.

## Admitted and excluded checkpoints

Admitted prompts are the checkpoint's declared unlabelled protein start, not a heading, instruction, or invented EC.

| Checkpoint | Native unlabelled prompt | Notes |
| --- | --- | --- |
| ProGen3-3B | `"1"` | Already complete under EXP-R2-233; do not regenerate |
| ProtGPT2 | `<EOT>` plus newline | No trailing `<EOT>` on the prompt |
| ProGen2-small / base / medium / large / xlarge | `"1"` | Medium's R227 200-attempt floor is not reused |
| ProGen3-112M | `"1"` | Same official generator as 3B; 3B does not substitute |
| ProteinGLM-7B-CLM | `<gmask><sop><eos>` | float32 |
| ProtGPT3-1.3B | `<\|bos\|>` then `"1"` | |
| RITA-xl | leading document `<EOS>` | No pad; one sequence at a time; cache off |
| Galactica 125M / 1.3B / 6.7B / 30B | `[START_AMINO]` | Not a `#` heading; protein FP32 protocol; 30B memory must be measured, not inferred from scoring |
| InstructProtein | tokenizer BOS plus `<protein>` | Not `Instruction:` |
| ProLLaMA Stage 1 / Stage 2 | `Seq=<` | Not `Superfamily=` |

Excluded: ZymCTRL (native input is an EC); every text-only checkpoint; Llama-2-7B (`Seq=<` is a scoring wrap, not a declared unlabelled generator). Galactica-125M protein context-information non-identification does not revoke the `[START_AMINO]` slot.

## Shared operating point

800 attempts, retained in full. Batches of eight, except RITA and any arm that refuses padding, which run one at a time. Batch seed `20260905 + batch_index`. Temperature 0.85, top-p 0.95, explicit top-k 50, repetition penalty 1, `min_new_tokens=0`, `max_new_tokens=400`, no perplexity post-selection. ProGen3's official API adds two delimiter positions (effective Hugging Face cap 402) and keeps cache on. ProGen2 rungs keep cache off.

Every attempt stores the raw continuation, the stop or compile status the interface actually exposes, a strict leading AA20 run, and empty or censored strings. Failures stay failures. `arm` is the checkpoint name, `condition=unconditioned`, `class_key=null`, `target_profile_hit=null`.

After the batch exists, draw 128 length-stratified structure parents by the EXP-R2-232 score-independent rule, or all supported sequences if fewer, each with one composition shuffle. Predictor, confidence event, and the shared natural calibration are those already attained in EXP-R2-232. The primary contrast is the design-weighted generation-minus-own-shuffle mean CA-pLDDT, with 4,000 near-duplicate sequence-group bootstrap resamples at seed 20260905. New arms use the reconciled UniRef50 search that reports identity and both coverages from the same hit.

Do not add samples, change decoding, or swap models after outcomes are seen. Interface failure of one arm is a named unavailable cell, not a silent replacement.

## Execution

L20 is for interface smokes only. Scientific generation waits until MegaScale Galactica-30B scoring and the stage-46 homologous-context expansion have released the current four-card allocation, or until `nvidia-smi` inside the selected pod shows an unused card. Do not take cards from those two running jobs.

The runner is `scripts/transfer/48_unconditional_generation.py`. The queue manifest is `scripts/transfer/campaign_s48_unconditional_generation.tsv`. Completion of an arm is: interface gate, 800 raw attempts, immutable ledger, structure subset to a terminal state, and the calibrated analysis. A complete computational package is not laboratory function.
