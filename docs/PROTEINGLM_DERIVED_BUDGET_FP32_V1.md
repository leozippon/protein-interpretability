# ProteinGLM derived-budget FP32 interface qualification v1

**Status:** protocol text for an independent controller. Not a freeze, not a run, not EXP-R2-225, not ProteinGym, and not panel admission. A later H200 cell must pin this source, bind runtime/receipts, and keep `experiment_admitted=false`.

Protocol id: `proteinglm-derived-budget-fp32-v1`.
Artefact: `proteinglm_derived_budget_qualification.json`.
CLI: `scripts/transfer/proteinglm_budget_qualification.py`.

## Why a new door

`second_stage_interface_qualification.py` remains the EXP-R2-225 sibling. Its ProteinGLM row stays in `DECLARED_UNAVAILABLE`. That campaign scores tokenizer defaults over `ids[1:]` (prefix and residue 1, and possibly a trailing EOS) at default bfloat16. Deleting the skip would measure the wrong estimand. This protocol does not edit that runner.

## Estimand

Production numbers come from one path only:

1. `load_arm_spec(arm_spec("proteinglm-7b-clm"), dtype="float32", strict=True)`
2. `Cohort.input_strings` → native prefix `<gmask><sop><eos>` = `[29, 32, 34]`, AA20 residues, no trailing EOS
3. `tokenize_batch` / `encode_budget_text`
4. `budget.scored_tokens` with `target_rule=residues_2_to_L`, shifted columns `q >= 3`, `L−1` targets, `prefix+L ≤ 1024`, `L ≥ 2`

That is continuation NLL, not a full-protein CLM and not a ProteinGym LLR. `scored_tokens` returns a float64 container; that is not an FP64 arithmetic claim. `config.max_length` is usually 20 from `PretrainedConfig` and is recorded, not used as the scoring window. Context is checkpoint `seq_length` 1024.

`strict_load` and `serving_provenance={derived,max_length}` are read from the Arm this call returned. They are not reconstructed by a later `derive_serving_package`.

Published qualification loads the declared 7B checkpoint only on CUDA. CPU must not load those weights. Tiny CPU helpers are not a `--tiny` flag and cannot mint `derived-budget-interface-qualified`.

## Frozen numeric cases

Fixed order. Residue 1 is never a target; the last residue is.

| name | sequence | L | targets |
|---|---|---|---|
| `minimum_two_residues` | `AC` | 2 | 1 |
| `canonical_aa20` | `amino_acids.AA20` | 20 | 19 |
| `avgfp_n80` | first 80 residues of avGFP, SHA-256 `3a6033d2eb88fa724c2aab48cb19d903201b1c2efefda1b82b8edb83a03ebbcf` | 80 | 79 |
| `longest_legal_1021` | `AA20` cycled and cut to `CONTEXT_LENGTH - PREFIX_LENGTH` = 1021 | 1021 | 1020 |

`avgfp_n80` copies the immutable EXP-R2-225 protein-probe string and hash only. It does not inherit that campaign's anagram, repeat ceiling, shuffle floor, dtype, or scoring window.

Production may call `scored_tokens` once on the four rendered strings with `batch_size=1` and `max_len=1024`. The independent reference is a separate shifted `torch.nn.functional.cross_entropy(reduction="none")` per case on the same ids/mask and the same eval model. It does not call `sequence_target_mask` and does not copy the production `log_softmax`+gather path. Hand-kept columns are `q >= 3` inside that row's length and valid mask. Per-target ids, indices, and counts must match production. Every compared loss must be finite. Max abs ≤ `precision_policy.FP32_PER_TARGET_ABS` (`1e-4`). Live logit width is the actual `[B,S,V]` last axis (128), not `lm_head` or config width used as a substitute. Parameters and reference logits are FP32. `torch.inference_mode`, disabled autocast, and `fp32_matmul_context` wrap both production and reference; the observed matmul/TF32 policy is recorded and restored.

No Spearman, no ProteinGym, no shared-premise capability claim, no shuffle-gain door, and no off-contract missing-prefix branch.

## Resource gate (published CUDA only)

After the 7B model is loaded and numeric local tensors have been released: synchronize, `reset_peak_memory_stats`, then enter the shared FP32 context. **One** public `scored_tokens` call on `[longest, longest, longest]` with `batch_size=1` and `max_len=1024` sits in a try/finally of its own: `attempted=1` is set immediately before that call, `completed=1` only if it returns. The finally block is the only terminal public-call telemetry: one bounded capture of duration and allocated/reserved peaks. That capture must not replace the public-call exception. Complete telemetry (`synchronize=="ok"` and finite required readings) is required before the extra independent reference and comparisons. A precision-context enter failure leaves attempted/completed at 0 and does not invent a public-call capture. Baseline fields are attached to progress first, then filled one by one.

`resource_public_call_count=1` and `finite=true` only after the resource gate succeeds. An OOM must not write those success marks. A single telemetry reading that fails is recorded as `unknown` plus a sanitised error; it is never invented as 0, and it must not replace the primary exception. `synchronize!="ok"` is incomplete telemetry and cannot mint `derived-budget-interface-qualified`. No extra VRAM percentage cap. No three independent public calls, no short-string substitute, no window shrink, no OOM fallback. Batch-1 qualification does not generalise other batch sizes. A failed run keeps the complete numeric case records already obtained, plus any `strict_load` / `serving_provenance` taken from this Arm, without re-deriving them after the fact.

## Verdict

`status` is only `derived-budget-interface-qualified` or `failed`. `experiment_admitted` is always false. `not_panel_admission` is always true. This is not EXP-R2-225 PASS and not ProteinGym admission.

Runtime in the artefact is a whitelist: Python, torch, transformers, CUDA version, and device. No hostname, pod name, environment, or argv. Expected published runtime is ct Python 3.11.14 / torch 2.9.1+cu128 / Transformers 4.57.3; snapshot/C_ENV/receipt binding is external.

Source identity hashes the controller, encoder, production scorer, and load/render modules actually used, plus the Arm's `derived` / `strict_load` records. It does not guess a neighbouring Git HEAD.

Any existing canonical artefact path is left unread and unreplaced, whether it is `failed`, `derived-budget-interface-qualified`, corrupt JSON, or any other directory entry. Publication writes a complete sibling temporary through shared `write_json`, then `os.link`s it into place; `os.link` does not replace. A collision or write error must not leave a half-written canonical file. Retry with a fresh `--out` or a new label; this controller does not back up, clear, or overwrite old records. After valid CLI arguments on a new output directory, a failure writes a sanitised `failed` JSON, exits non-zero, and does not mint a qualified status. The 7B checkpoint has not been run under this protocol.

Training, generate, and cache reuse are unsupported. Gradient checkpointing raises; ordinary `.train()` without checkpointing may still forward. JIT import flags from derived modeling remain a process-wide side effect; a published cell should be independent.
