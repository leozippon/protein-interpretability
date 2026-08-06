# Project Status

## Current State

InterpretabilityTransfer is the only live research programme in this repository. Its objective is to compare text and protein generative models, measure how existing interpretability methods transfer, and design a protein-adapted method only after a concrete transfer limitation is established.

`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is authoritative for scientific claims, limitations, retractions, and the phased plan. Nothing below is a second source for any of them; where this file and that one disagree, that one wins.

**The foundational programme closed on 2026-08-06.** Its results are frozen as claims F1–F9 in audit §1.1, and re-measuring any of them is not authorised. The main line is now **D2.g**: audit the external state of the art rather than add another draw of an instrument this repository has already characterised twice.

## Progress

- **Model comparison (part 1):** closed to open-ended measurement. The surviving induction result is a probe-dependent upper-tail difference, not a general modality or capability claim. The only citable part-1 statement is the pathway budget (F7).
- **Method transfer (part 2):** the deliverable, and where the yield is. Twenty-three limitations are catalogued, several demonstrated on text controls and therefore properties of the **method** rather than of protein models. Two results carry the position. On copy suppression, a prediction-addressed-attention census does not recover its own causally important heads on any residue-tokenised protein decoder measured, while every text decoder clears 3.6× its own chance — with architecture, conditioning, scale, corpus and symbol-level tokenisation each excluded by a control built to exclude it (F1, F2). On induction, the analogous separation is depth-confounded on every arm including the text controls, so it is a method limitation (F5).
- **Root cause of the copy-suppression failure:** measured and **negative** (F3). No instance-level factor accounts for it, and the "causal target is noise" account is falsified rather than merely unsupported. What arrived instead is F1b: a selector that knows only a head's layer index beats the census on ProGen2-small, and carries most of ProtGPT2's apparent pass. A protein-adapted selector (D3.e) is therefore **dead**, not deferred.
- **External baseline (D2.g), active:** ProGenMech (arXiv:2606.16044) trains cross-layer transcoders on ProGen3-112M and reports sparse circuits recovering that model's generative distribution and zero-shot fitness. Their CLT weights are unobtainable, so this audit gates their **baseline, not their headline**, and that qualification travels with every result. Under sequential replacement the released PLT recovers 12.6–13.2% of the clean-to-fully-ablated NLL gap, fails the causal ranking gate in all six family × run cells, and does so with attainability passing at ceilings of 0.984–0.994 (EXP-R2-132).
- **Adapted method (part 3):** not yet earned. Construction stays gated on a reproducible root cause. Methods are to be developed for dense models first, then for MoE, and unified only afterwards.

The active contract's arm and stage counts live in `scripts/transfer/panel_contract.py`, which is the scheduling source of truth; never restate them by hand. `python scripts/transfer/panel_contract.py --json` prints them.

## Remaining Work

D2.g, in the order the evidence requires:

1. **Score the base against what is free — done (EXP-R2-133, bounded by EXP-R2-134).** Standing rule 28 requires it. On ProGenMech's eight assays ProGen3-112M's zero-shot fitness is not separable from a BLOSUM62 lookup under either sampling design (+0.066 [−0.110, +0.241]; +0.045 [−0.099, +0.190]). **Across the full 217-assay benchmark it is** — +0.0647 [+0.0386, +0.0909], winning 143 of 217. The effect size is the same on both cohorts and only the power differs, so the finding is **not** that the model is at baseline level; it is that their eight-assay panel cannot resolve the advantage their recovery ratios are quoted against. A ratio on a base whose own footprint is unresolvable on its own panel cannot separate a circuit that captured the model's fitness computation from one that captured a substitution matrix — L1's shape, measured.
2. **Explain the residual gap to ProteinGym's published score for the identical checkpoint** (0.497 over six of the same assays). Our reproduction agrees with *their* 0.29 base under both sampling designs, so the sampling design is **not** the explanation — an inference this repository briefly held and its own uniform-draw arm falsified. The two largest per-assay discrepancies are the two most mutation-dense assays, which is where a difference in how multi-mutant variants are scored would show first. Their "likelihood recovery" is separately a sample-quality ratio between different sequences, not a fixed-cohort recovery.
3. **Corpus correspondence** for the behavioural result: our reconstruction NMSE sums to 4.42 against the 3.54 their checkpoint records, with a bit-identical forward.
4. **Whether routing explains the replacement's failure.** Their CLT abstracts the MoE router away by construction and says so. Whether the error concentrates where routing is unusual is a within-model question, needs no second MoE, and is motivated by a measured failure rather than by a category expectation.

Do not revive retired steering, atlas, drug-design, encoder-benchmark, or venue-specific plans.

## Storage

Compact evidence is under `evidence/`. Large data, checkpoints, generated results, and logs remain outside Git. Frozen history and retired project material were checksum-verified and moved to `/Data2/lzp/bio_archive`; see `docs/ARCHIVE.md`.

Historical status snapshots were preserved in that external archive before this file was reduced to current operational state.
