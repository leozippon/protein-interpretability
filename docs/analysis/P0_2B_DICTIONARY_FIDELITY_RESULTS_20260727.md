# P0-2 and P0-2b dictionary-fidelity results

**Date:** 2026-07-27 **Scope:** completed P0-2 adjudication and prospective P0-2b/P1 downstream instrument qualification

## Result

The original P0-2 panel failed its frozen gate. Only ProGen2-medium TopK was atlas-eligible; this does not make the whole panel eligible.

The separate prospective P0-2b qualification also returned no eligible sparse model/method. This does not modify the original receipt. It means none of the completed checkpoints may enter P2–P8 under the 2026-07-27 downstream decision rule.

## Original frozen P0-2 adjudication

The executed gate specification has SHA-256 `f96aa1da3316adc064aa9cb14e03c772e8dc3b3954325ba3387a61530f8af2a1`. The eligibility receipt has SHA-256 `3f472afb0171836ad7f51d5c1d9e25b1d8d4a67aae1ac1aa51744bf18494f9b3` and panel status `one_or_more_model_method_quality_gates_failed`.

| Model | Method | Frozen P0-2 outcome |
|---|---|---|
| ProtGPT2 | TopK | failed median dead-fraction gate in every seed |
| ProtGPT2 | ReLU/L1 | terminal validation sparsity-match failure; no full runs |
| ProtGPT2 | gated | terminal validation sparsity-match failure; no full runs |
| ProtGPT2 | dense | failed required-layer FVU count in every seed |
| ZymCTRL | TopK | failed dead-fraction gate in every seed; seed 17 also failed the required-layer count |
| ZymCTRL | ReLU/L1 | failed required-layer and dead-fraction gates in every seed |
| ZymCTRL | gated | failed required-layer and dead-fraction gates in every seed |
| ZymCTRL | dense | failed mean FVU and required-layer gates in every seed |
| ProGen2-medium | TopK | **atlas-eligible in all three seeds** |
| ProGen2-medium | ReLU/L1 | terminal validation sparsity-match failure; no full runs |
| ProGen2-medium | gated | failed dead-fraction gate in every seed |
| ProGen2-medium | dense | failed required-layer FVU count in every seed |

The executed specification contains 27 completed full runs. The three sparsity-terminal model/methods account for the nine intentionally absent full runs.

## Prospective P0-2b design

The protocol was `docs/P0_2B_DICTIONARY_FIDELITY_PROTOCOL_20260727.md`; it moved to `../../../archive/legacy/r2_retired_scope_20260729/docs/P0_2B_DICTIONARY_FIDELITY_PROTOCOL_20260727.md` on 2026-07-29 (EXP-R2-065) with the rest of the P0 protocol set. The executed specification, which is the authoritative form, has SHA-256 `11092e16891922380a2d224288138968e45ea982d36b46a4007f9f6b76a8d6bf` and is at `../../evidence/p0_2b_fidelity_20260727/p0_2b_fidelity_spec.executed.json`.

The new cohort contains 240 canonical-AA EC-labelled Swiss-Prot sequences, 64–246 residues, selected from 248 eligible sequences after excluding:

- all six original P0-2 train/validation/test manifests;
- the first 40,000 ZymCTRL source records, covering the July 24 pilot source prefixes; and
- duplicate and noncanonical sequences.

Its JSONL SHA-256 is `19474c1f94be9e4e3ff9b9ef30c76b5870a211b1c8dc3b7d0c0852afebc35dd3`. Tokenizer-only preflight found maximum native-input lengths of 85 tokens for ProtGPT2, 256 for ZymCTRL and 247 for ProGen2-medium. No input was truncated.

At relative depth 0.5, the evaluation compared clean, exact reinjection, training-target-mean ablation and dictionary reconstruction. The prespecified qualification required, in every seed:

- mean-ablation CE delta at least 0.05 nats/token;
- mean-ablation KL at least 0.01 nats/token;
- loss recovered at least 0.80;
- KL recovered at least 0.80; and
- exact reinjection.

## P0-2b results

All 27 runs passed the reinjection check: maximum logit difference was exactly zero, argmax agreement was 1.0 and the CE tolerance passed.

| Model | Method | Mean-ablation CE delta | Mean-ablation KL | Loss recovered, seeds 17/29/43 | KL recovered, seeds 17/29/43 | Qualification |
|---|---|---:|---:|---|---|---|
| ProtGPT2 | dense | 0.033783 | 0.161389 | undefined | undefined | fail: CE denominator |
| ProtGPT2 | TopK | 0.033783 | 0.161389 | undefined | undefined | fail: CE denominator |
| ZymCTRL | dense | 0.015783 | 0.025140 | undefined | undefined | fail: CE denominator |
| ZymCTRL | gated | 0.015783 | 0.025140 | undefined | undefined | fail: CE denominator |
| ZymCTRL | ReLU/L1 | 0.015783 | 0.025140 | undefined | undefined | fail: CE denominator |
| ZymCTRL | TopK | 0.015783 | 0.025140 | undefined | undefined | fail: CE denominator |
| ProGen2-medium | dense | 0.061427 | 0.083770 | 0.120 / 0.114 / 0.102 | 0.071 / 0.060 / 0.080 | fail |
| ProGen2-medium | gated | 0.061427 | 0.083770 | 0.323 / 0.305 / 0.331 | 0.229 / 0.234 / 0.243 | fail |
| ProGen2-medium | TopK | 0.061427 | 0.083770 | 0.368 / 0.355 / 0.333 | 0.262 / 0.254 / 0.250 | fail |

The strongest run was ProGen2-medium TopK seed 17. Its 1,000-replicate sequence-cluster bootstrap interval was `[0.323, 0.411]` for loss recovered and `[0.240, 0.282]` for KL recovered. Even the upper bounds are far below 0.80.

ProtGPT2 and ZymCTRL ratios are intentionally undefined. Their positive raw denominators must not be used to construct post-hoc ratios after failing the prespecified CE-denominator guard.

## Relation to FVU

On the only model with valid recovered metrics, lower P0-2 test FVU ranked methods in the same direction as higher behavioural fidelity:

| ProGen2-medium method | Mean test FVU | Mean loss recovered | Mean KL recovered |
|---|---:|---:|---:|
| TopK | 0.304261 | 0.351909 | 0.255253 |
| gated | 0.331368 | 0.319493 | 0.235308 |
| dense | 0.499447 | 0.112083 | 0.070282 |

Across the nine seed-level ProGen2-medium runs, descriptive Spearman correlations were `rho = -0.867` for FVU versus loss recovered and `rho = -0.850` for FVU versus KL recovered. Lower FVU is better, so these negative correlations indicate agreement, not an inverted ranking. The nine runs contain only three method clusters and are not independent evidence for a general law, but they do not support extending TG-03/TG-07's exploratory “FVU ranks protein dictionaries backwards” claim to the production P0-2 panel.

## Interpretation and next decision

1. P0-2 remains a panel-level negative under its original procedure.
2. The single-layer P0-2b intervention is too weak to define the prespecified recovered ratios for ProtGPT2 and ZymCTRL. This is a site/denominator failure, not evidence that their dictionaries lack biological features.
3. ProGen2-medium has valid denominators and reproducible partial recovery, with TopK better than gated and dense, but no method approaches the downstream qualification threshold.
4. P0-2b contains no production matched-text dictionary. It cannot establish a text-versus-protein transfer mechanism or explain why Anthropic's programme works in text.
5. Under the frozen decision rule, do not launch P2–P8, legacy P0-3–P0-8, steering or atlas expansion. A future attempt would require a new protocol with a matched text instrument and a behaviorally consequential, denominator-valid replacement design, such as a prespecified multi-layer replacement. That work was not started here.

## Integrity, compute and storage

- Aggregate SHA-256: `7f68e08775af87a171f2e4aac2a0d88cf235280280bf1bac44f76b0a2959bd07`.
- Evidence archive SHA-256: `06006fd31728c73f9a4412637153db2ec227b2a827698249f87984d99b4feac4`.
- Code archive revision 5 SHA-256: `2dcbbaf3405e92bf9db50c2c5c2ae43f789c7ee119dfa0ddd3e786da2eaf5a5e`.
- Queue wall times: 138.039, 228.252, 137.093 and 267.383 seconds.
- Sum of queue wall time: 770.767 accelerator-seconds, or 0.2141 accelerator-hours.
- Sum of per-run evaluation time: 208.914 seconds.
- Maximum allocated accelerator memory: 14,242,418,176 bytes (13.264 GiB).
- Compact result JSON: 5,880,475 bytes; synchronized evidence tree: 6,045,859 bytes.
- Post-run GPUs were idle. GPFS remained 21% used with about 44 TB available. No H200 issue required user action.
- Local verification passed 241 tests plus 6 subtests; the targeted fidelity suite passed 18 tests. Ruff check and format checks passed.

The retained P0-2 checkpoints and exact caches were not deleted: they remain bound scientific inputs and may be required to reproduce this result or design a separately approved follow-up. The new P0-2b outputs are compact and do not create material storage pressure.
