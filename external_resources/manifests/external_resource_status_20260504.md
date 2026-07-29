# External Resource Status (2026-05-04 CST)

H200 root: `/oss-pvc/zhk_zip/biocc/external_resources`

## Ready / Staged

| Resource | Status | Local path | H200 path |
|---|---|---|---|
| HMMER 3.4 | compiled on H200 | `external_resources/tools/hmmer-3.4.tar.gz` | `tools/bin/hmmscan`, `tools/bin/hmmsearch`, `tools/bin/hmmpress` |
| Pfam-A HMM | downloaded, decompressed, hmmpressed | `external_resources/ec_metrics/pfam/Pfam-A.hmm.gz` | `ec_metrics/pfam/Pfam-A.hmm` + `.h3*` index |
| DIAMOND 2.1.24 | ready | `external_resources/tools/diamond-linux64-v2.1.24.tar.gz` | `tools/bin/diamond` |
| Foldseek AVX2 | binary ready, target DB missing | `external_resources/tools/foldseek-linux-avx2.tar.gz` | `tools/bin/foldseek` |
| AlphaMissense | staged and scored for local R1 pathogenicity | `external_resources/baselines/alphamissense/*.tsv.gz` | `baselines/alphamissense/*.tsv.gz` |
| CLEAN | source staged, weights missing | `external_resources/ec_metrics/clean/CLEAN_source_20260504.tar.gz` | `ec_metrics/clean/CLEAN` |

## Remaining Blockers

- PrimateAI-3D: gated Hugging Face dataset; requires accepting Illumina academic license and approved access.
- gMVP: available through dbNSFP, but dbNSFP download/licensing path needs user-approved acquisition.
- ESM-1v: not downloaded yet; choose HF Transformers weights vs FAIR ESM checkpoints before staging the large files.
- CLEAN pretrained package: Google Drive pretrained CLEAN bundle and ESM-1b weights are still missing.
- Foldseek target DB: binary is ready, but no bounded target structure database has been selected or built.

## H200 Usage

```bash
source /oss-pvc/zhk_zip/biocc/external_resources/setup_h200_external_env.sh
diamond version
hmmscan -h | head
foldseek version
```

## Completed Follow-Up Runs

- AlphaMissense baseline scored on R1 ClinVar2000 and CancerHoldout101: see `Research1/results/variant_effect/alphamissense_baseline_20260504.md`.
- Pfam/HMMER scan completed for R2 lysozyme generated sequences: see `Research2/results/ec_metrics/pfam_generated_lysozyme_20260504.md`.
