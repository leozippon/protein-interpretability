# Project Status

## Current State

InterpretabilityTransfer is the only live research programme in this repository. Its objective is to compare text and protein generative models, measure how existing interpretability methods transfer, and design a protein-adapted method only after a concrete transfer limitation is established.

`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is authoritative for scientific claims, limitations, retractions, and the phased plan.

## Progress

- **Model comparison:** closed to open-ended measurement. The surviving induction result is a probe-dependent upper-tail difference, not a general modality or capability claim.
- **Method transfer:** active and the main deliverable. Twenty-three limitations are catalogued; several were demonstrated on text controls and therefore constrain the method rather than protein models. The strongest directly measured result is L22 (EXP-R2-071/072/077/079/081): with every head patched, an all-grid rank correlation between a head-prevalence census score and the causal effect separates text from protein decoders. The corpus-draw and cross-lab checks both landed. **EXP-R2-120 then narrowed it and the narrowing is the current headline:** that correlation is confounded with layer depth on every arm, and depth-controlled the separation falls to +0.55–1.39 pooled boundary-arm standard deviations against +5.08–7.18 raw — surviving in sign, not as a modality claim. L22 is therefore now scoped as a **method** limitation.
- **Copy-suppression replication (D2.c):** under re-measurement. EXP-R2-116's layout correction was applied to the causal side only — the census score is a per-sequence aggregate and cannot be filtered after a run — so its withdrawals and its "strengthened separation" are themselves withdrawn (EXP-R2-119). The guard is repaired at the root (EXP-R2-118) and the rebuilt pools are being taken to K≥3 by EXP-R2-121.
- **Adapted method:** not yet justified. Construction remains gated on a reproducible root cause from the transfer evaluation.

The active contract contains 12 model arms and 11 stages. `scripts/transfer/panel_contract.py` is the scheduling source of truth; never restate its counts by hand.

## Remaining Work

Run the powered method-transfer stages defined in audit Phase B, evaluate them against frozen gates, and launch a Phase C construction experiment only if a measured failure mode supports it. Do not revive retired steering, atlas, drug-design, encoder-benchmark, or venue-specific plans.

## Storage

Compact evidence is under `evidence/`. Large data, checkpoints, generated results, and logs remain outside Git. Frozen history and retired project material were checksum-verified and moved to `/Data2/lzp/bio_archive`; see `docs/ARCHIVE.md`.

Historical status snapshots were preserved in that external archive before this file was reduced to current operational state.
