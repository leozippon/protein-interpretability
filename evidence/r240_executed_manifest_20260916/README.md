# Executed EXP-R2-240 manifest

This directory preserves the campaign manifest that actually produced the published EXP-R2-240 context-information identification table. `campaign_r240_extra_context.executed.tsv` is a byte-for-byte copy of the manifest found in the executed run's GPFS snapshot. The manifest committed in this repository at `scripts/transfer/campaign_r240_extra_context.tsv` is **not** that manifest: it belongs to a superseded earlier dispatch of the same campaign.

## Executed run

- Run-id `20260916033459_68af0c6949cd`, working-tree freeze (not `--pin`).
- Snapshot `packages/20260916033459_68af0c6949cd` under the GPFS project root `/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer`.
- Queue detached 2026-09-16T15:13:59Z and closed 2026-09-16T16:47:20Z on the 4-GPU allocation.
- Stage 01 tally: 192 `exited-ok`, `FAILURES 0`, `NO-RECORD 0`.
- Stage 41 report for this run is `results/external_baseline/20260916033459_68af0c6949cd/r240_stage41/`, containing `DONE`, `context_information_bootstrap.json`, `identification_table.json` and `inventory.json` with `n_sidecars=192`. No `r240_stage41` directory exists under the superseded run's tree, which is how the published numbers were traced to this run.

## Hashes

- Executed manifest (this directory, `campaign_r240_extra_context.executed.tsv`): `cc2e6b4ef11d09000cb21554fd9eeee06af8276a637fc3dc90a06d36655bcbce`, 205 lines, 99,827 bytes.
- Committed manifest (`scripts/transfer/campaign_r240_extra_context.tsv`): `8575a7dc6e92cc15c4a560a282d3f42ed5809bbd1138d1d4c9750314abeded34`, 205 lines, 99,838 bytes. It is byte-identical to the manifest of the superseded first dispatch for run-id `20260916034042_b1cd841c08ef`.

## Superseded first dispatch

EXP-R2-240 was dispatched twice on the same 4-GPU allocation. The first dispatch, run-id `20260916034042_b1cd841c08ef`, also ran to completion: its 192 cell logs span 2026-09-16T10:48Z–12:23Z and its 192 artefacts sit under `results/external_baseline/20260916034042_b1cd841c08ef/` on GPFS. That run's own status file is not preserved at the queue runner's default path `<gpfs>/logs/external_baseline/campaign_r240_extra_context.status.tsv`, because both dispatches derive that path from the same manifest basename and the second dispatch overwrote it.

## Measured equivalence

The two dispatches differ only in scheduling. All of the following were measured:

- The 192 labels are the same set; there are no labels unique to either manifest.
- All 192 `(label → stage, env, expect, args)` tuples are identical, including targets, modes, cohorts, seeds, skips, block counts and expected JSON basenames.
- Only `(slot, gpu)` assignment differs. The executed manifest packs four distinct arm groups into each slot across `cuda:0..3`; the committed manifest assigns four blocks of a single arm group to each slot. 180 of the 192 labels move; 12 keep their assignment.
- The three sampled power JSON pairs (`galactica-30b_protein` b0, `progen2-large` b7, `qwen3-8b-base` b0) are identical after removing `created_utc` and `settings.device`, and their `settings.device` values differ between the runs (`cuda:0` versus `cuda:3`).
- Sampled `.records.npz` files from the two runs are byte-identical.

Because every cell's stage, arguments and expected output are unchanged and the observed statistics agree, no published number, interval, verdict or conclusion changes. The defect is provenance wording plus an uncommitted executed manifest, not a data defect.

## Executed code reconstruction

The executed snapshot's `CODE_CONTENT_SHA256SUMS` records 186 file hashes. Commit `7a06558` reproduces 184 of them. The two exceptions are `scripts/transfer/campaign_r240_extra_context.tsv` itself, which differs by construction, and `scripts/transfer/20_retrieval_bound.py`, whose frozen hash `8ea90e796b154367fe35f2e27b669fd62182a98340b6c5eb74c1d6a17f805a70` equals the version at commit `2de539a2`. Stage 01 imports no `20_*` entry point, so neither exception reaches the code that produced the stage 01 sidecars or the stage 41 table.

## Sources

The facts above were established by a read-only audit of the pod-side GPFS snapshots on 2026-09-16 and by re-verification on the transfer host. Specifically:

- The executed manifest's content and `cc2e6b4e…` hash come from the executed snapshot, whose own `CODE_CONTENT_SHA256SUMS` records the same hash for that path; the copy here was re-hashed after copying.
- The committed manifest's `8575a7dc…` hash, the byte-identity with the first dispatch's manifest, and the label/args equivalence were measured with `sha256sum`, `cmp`, `wc -lc` and a field-by-field comparison of the two TSVs on the transfer host.
- The run-ids, the 15:13:59Z–16:47:20Z window, the 192 `exited-ok` / `FAILURES 0` / `NO-RECORD 0` tally, the `r240_stage41` contents and the absence of `r240_stage41` under the first dispatch are audit observations of the GPFS snapshot trees; the 192 cells equally confirm the first dispatch through its logs and artefacts.
- The power-JSON and `.records.npz` comparisons are audit observations; the three power JSON pairs were re-checked on the transfer host.
- The 184/186 hash reconstruction and the `2de539a2` identity of `20_retrieval_bound.py` were recomputed on the transfer host with `git show <rev>:<path> | sha256sum`.

Neither run-id tree is mirrored into this worktree's ignored `results/transfer/external_baseline/` directory, so the GPFS-side observations above are attributed to the audit rather than re-derived here.

Verify this package from this directory with `sha256sum -c SHA256SUMS`.
