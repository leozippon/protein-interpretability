# P0-2 mask-validation evidence (2026-07-21)

This directory holds the durable confirmatory receipt for the exact dictionary
training module used by the bfloat16 screening lineage and required unchanged
for the future full-run lineage.

## Production command

Run from the repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python scripts/76_write_mask_validation_receipt.py \
  --output evidence/p0_2_mask_validation_20260721/mask_validation_receipt.json \
  --expected-module-sha256 347a095c2e18a429e09011f84a45bd40b1cf5d46ebf38398e6e2a7afe57c6596
```

The command exited 0. Both frozen pytest nodes passed, and the receipt reports
`production_scientific_eligibility: true`. Its module and test descriptors are
relative to the receipt, so the existing gate resolves the same files on either
repository mirror.

## Immutable identities

- receipt SHA-256:
  `5966e274881984b2eeabeedd749d94c313fce02fb814911f0d1082ac3c3232db`
- tested `src/revision/dictionary_controls.py` SHA-256:
  `347a095c2e18a429e09011f84a45bd40b1cf5d46ebf38398e6e2a7afe57c6596`
- tested `tests/test_dictionary_controls.py` SHA-256:
  `a1f2f61f8933c1e1a766bc5e827ba3747215c2286a2df2421e32d0e2b8fc1253`
- producer `src/revision/mask_validation_receipt.py` SHA-256:
  `de9324909af75014a0bf76420abf5a1904047891b12c5da595dc9c96836180c0`
- CLI `scripts/76_write_mask_validation_receipt.py` SHA-256:
  `287844fa0e88061aa167cce1fa1233428c3289cd26123f76463c9eb88d17ce68`

Direct validation with
`src.revision.dictionary_gate._validate_mask_receipt` returned `True`, the
tested module digest above, and the expected test-file digest. This receipt is
necessary P0-2 evidence only; it does not by itself pass the dictionary-quality
gate or authorize downstream biological claims.
