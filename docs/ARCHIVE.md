# External Archive

Frozen provenance was moved from the live repository to `/Data2/lzp/bio_archive` on 2026-07-29. The external tree contains retired R0/R1 work, the retired sparse-readout and manuscript scope, historical result snapshots, conversation and planning records, old configurations, the retired H200 sync script, and preserved nested Git histories.

The move used copy-then-verify semantics. Source and destination matched at 2,907 regular files and 4,825,760,628 bytes before source removal. Retired live files increased the final set to 2,928 regular files. `/Data2/lzp/bio_archive/MIGRATION_SHA256SUMS` verifies every regular file, and both preserved bare Git histories passed `git fsck --full`; one repository reports a harmless dangling blob.

The archive is immutable provenance, not a runtime dependency. Historical paths and names are intentionally not normalized. New work belongs in the live repository, and a required historical artifact should be promoted as a compact receipt under `evidence/` rather than edited in place.
