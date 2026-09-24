#!/usr/bin/env bash
set -euo pipefail

# CPU-only Pfam + UniRef50 annotation for EXP-R2-246 ledgers.
# Does not take a GPU. Do not launch through the ESMFold2 campaign lock.
# Tool paths must already be staged; this script never installs or builds assets.

GPFS="${TRANSFER_PROJECT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer}"
PY="${TRANSFER_PYTHON:-${GPFS}/runtimes/ct-20260905/bin/python}"
CODE="${S48_ANNOTATE_CODE:-${GPFS}/packages/s48_annotate_native}"
MAIN_RUN="${S48_GENERATE_RUN:-${GPFS}/results/external_baseline/20260918194031_e37ee9cdeffc}"
RETRY_RUN="${S48_GENERATE_RETRY:-${GPFS}/results/external_baseline/20260918223307_9935857a1d58}"
OUT_ROOT="${S48_ANNOTATE_OUT:-${GPFS}/results/external_baseline/s48_annotate_native}"
ASSETS="${S48_ANNOTATION_ASSETS:-${GPFS}/data/r233_annotation_assets}"
HMMSCAN="${S48_HMMSCAN:-${ASSETS}/bin/hmmscan}"
PFAM="${S48_PFAM_HMM:-${ASSETS}/pfam/Pfam-A.hmm}"
DIAMOND="${S48_DIAMOND:-${ASSETS}/bin/diamond}"
DMND="${S48_DIAMOND_DB:-${ASSETS}/reference/uniref50_full.dmnd}"
META="${S48_REFERENCE_METADATA:-${ASSETS}/reference/reference_corpus_metadata.json}"
THREADS="${S48_ANNOTATE_THREADS:-4}"
SHARDS="${S48_ANNOTATE_SHARDS:-2}"

ARMS=(
  progen3-112m galactica-125m progen2-small protgpt2 progen2-base
  progen2-medium rita-xl protgpt3-1.3b galactica-1.3b instructprotein
  progen2-large progen2-xlarge galactica-6.7b proteinglm-7b-clm
  prollama-stage-1 prollama galactica-30b
)

ledger_for() {
  local arm="$1"
  local candidate
  for candidate in \
    "${RETRY_RUN}/s48_generate_${arm}/attempts.jsonl" \
    "${MAIN_RUN}/s48_generate_${arm}/attempts.jsonl"
  do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

export CUDA_VISIBLE_DEVICES=
mkdir -p "${OUT_ROOT}"
for arm in "${ARMS[@]}"; do
  out="${OUT_ROOT}/s48_annotate_${arm}"
  if [[ -f "${out}/annotation_manifest.json" ]]; then
    echo "SKIP ${arm}: already annotated"
    continue
  fi
  if ! ledger="$(ledger_for "${arm}")"; then
    echo "MISSING_LEDGER ${arm}" >&2
    exit 2
  fi
  echo "ANNOTATE ${arm} <- ${ledger}"
  mkdir -p "${out}"
  PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PY}" "${CODE}/scripts/transfer/annotate_generation_evidence.py" native \
      --attempts "${ledger}" \
      --out "${out}" \
      --hmmscan "${HMMSCAN}" \
      --pfam-hmm "${PFAM}" \
      --diamond "${DIAMOND}" \
      --diamond-db "${DMND}" \
      --reference-metadata "${META}" \
      --threads "${THREADS}" \
      --shards "${SHARDS}"
done
echo S48_ANNOTATE_DONE
