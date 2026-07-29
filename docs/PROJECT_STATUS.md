# Project Status

## Current State

InterpretabilityTransfer is the only live research programme in this repository. Its objective is to compare text and protein generative models, measure how existing interpretability methods transfer, and design a protein-adapted method only after a concrete transfer limitation is established.

`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is authoritative for scientific claims, limitations, retractions, and the phased plan.

## Progress

- **Model comparison:** closed to open-ended measurement. The surviving induction result is a probe-dependent upper-tail difference, not a general modality or capability claim.
- **Method transfer:** active and the main deliverable. Nineteen limitations are catalogued; several were demonstrated on text controls and therefore constrain the method rather than protein models.
- **Adapted method:** not yet justified. Construction remains gated on a reproducible root cause from the transfer evaluation.

The active contract contains 11 model arms and 11 stages. `scripts/transfer/panel_contract.py` is the scheduling source of truth.

## Remaining Work

Run the powered method-transfer stages defined in audit Phase B, evaluate them against frozen gates, and launch a Phase C construction experiment only if a measured failure mode supports it. Do not revive retired steering, atlas, drug-design, encoder-benchmark, or venue-specific plans.

## Storage

Compact evidence is under `evidence/`. Large data, checkpoints, generated results, and logs remain outside Git. Frozen history and retired project material were checksum-verified and moved to `/Data2/lzp/bio_archive`; see `docs/ARCHIVE.md`.

Historical status snapshots were preserved in that external archive before this file was reduced to current operational state.
