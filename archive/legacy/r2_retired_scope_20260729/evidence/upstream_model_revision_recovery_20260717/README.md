# Upstream model revision recovery

This directory records immutable upstream-revision evidence recovered from the
deployed pretrained-model trees on 2026-07-17. It is provenance evidence, not a
public model-weight deposit and not a scientific result.

`protgpt2_manifest.json` verifies the complete deployed `nferruz/ProtGPT2`
tree at Hugging Face commit
`f71aa6cf063ad784ebd53881d11332fd098eaa58`. Every Hugging Face download
metadata file names that same full revision. For ordinary files, the recorded
ETag equals `git hash-object` of the deployed bytes; for the LFS model weight,
the recorded ETag equals the deployed file SHA-256. The manifest also embeds
each three-line metadata file as base64 so its original CRLF bytes can be
reconstructed and checked against the recorded metadata SHA-256 after the
compute copy is unavailable.

The upstream commit history identifies the corresponding verified abbreviated
commit `f71aa6c`:

- <https://huggingface.co/nferruz/ProtGPT2/commits/main>
- <https://huggingface.co/nferruz/ProtGPT2/commit/f71aa6cf063ad784ebd53881d11332fd098eaa58>

`zymctrl_manifest.json` records a narrower result for the deployed ZymCTRL
tree. Commit `3c532ef172b9cd2e95238baadf5167ebb89fbc32` is the best-supported
upstream snapshot and the deployed weight is cryptographically verified, but
strict whole-tree proof is incomplete because independent upstream object
identities were not recoverable for every ordinary file. The manifest must not
be read as an exact complete-tree revision receipt.

`progen2_medium_manifest.json` records that the deployed ProGen2-medium tree
does not match any single upstream commit. Its `config.json` combines a later
`_name_or_path` with the earlier unqualified `auto_map`, a combination absent
from the six-commit upstream history. File-level component provenance is
retained where recoverable, but exact reproduction requires depositing the
deployed local snapshot and its manifest rather than naming a nonexistent
whole-tree revision.

The P0-1 release gate remains open for the ZymCTRL whole-tree uncertainty, the
ProGen2-medium local snapshot, remaining raw artifacts, an approved tag and
DOI-backed licensed deposits.
