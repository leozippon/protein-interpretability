# Project Status

## Current State

InterpretabilityTransfer is the only live research programme in this repository. Its objective is to compare text and protein generative models, measure how existing interpretability methods transfer, and design a protein-adapted method only after a concrete transfer limitation is established.

`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is authoritative for scientific claims, limitations, retractions, and the phased plan.

## Progress

- **Model comparison:** closed to open-ended measurement. The surviving induction result is a probe-dependent upper-tail difference, not a general modality or capability claim.
- **Method transfer:** active and the main deliverable. Twenty-three limitations are catalogued; several were demonstrated on text controls and therefore constrain the method rather than protein models. The strongest directly measured result is L22 (EXP-R2-071/072/077/079/081): with every head patched, an all-grid rank correlation between a head-prevalence census score and the causal effect separates text from protein decoders. The corpus-draw and cross-lab checks both landed. **EXP-R2-120 then narrowed it and the narrowing is the current headline:** that correlation is confounded with layer depth on every arm, and depth-controlled the separation falls to +0.55–1.39 pooled boundary-arm standard deviations against +5.08–7.18 raw — surviving in sign, not as a modality claim. L22 is therefore now scoped as a **method** limitation.
- **Copy-suppression replication (D2.c):** answered and now the programme's live result. Both layout guards are repaired at the root (EXP-R2-118, EXP-R2-124) after EXP-R2-116's one-sided correction was itself withdrawn (EXP-R2-119). The eleven-arm panel completed at K=3–5 (EXP-R2-126): what survives the cross-lab control is a **within-arm** statement — a prediction-addressed-attention census on a residue-tokenised protein decoder does not recover its own causally important heads above its own chance level, where every text decoder exceeds its own by ≥3.6×. **EXP-R2-128 (ZymCTRL, K=4) rules out architecture and conditioning**; **EXP-R2-129 (ByGPT5-medium, K=4) then rules out symbol-level tokenisation** — a byte-level *text* decoder on an identical 192-head grid retrieves at 5.3× its own chance against ProGen2-small's 0.5× — so the failure is a property of the protein arms rather than of their tokenizer. The same control shows symbol granularity collapsing the all-grid rank correlation without impairing retrieval, which is why neither rank-correlation separation carries a claim; they also do not survive the cross-lab arms.
- **Adapted method:** not yet justified. Construction remains gated on a reproducible root cause from the transfer evaluation.

The active contract's arm and stage counts live in `scripts/transfer/panel_contract.py`, which is the scheduling source of truth; never restate them by hand.

## Remaining Work

The one measurement that can still decide what the D2.c result is *about* is a **byte-level text decoder** in the PAA census (EXP-R2-129): every symbol-level-tokenised arm in the panel is a protein model, so tokenisation granularity and modality cannot be separated within it. Beyond that, run the powered method-transfer stages defined in audit §9 D2, evaluate them against frozen gates, and launch a D3 construction experiment only if a measured failure mode supports it. Do not revive retired steering, atlas, drug-design, encoder-benchmark, or venue-specific plans.

## Storage

Compact evidence is under `evidence/`. Large data, checkpoints, generated results, and logs remain outside Git. Frozen history and retired project material were checksum-verified and moved to `/Data2/lzp/bio_archive`; see `docs/ARCHIVE.md`.

Historical status snapshots were preserved in that external archive before this file was reduced to current operational state.
