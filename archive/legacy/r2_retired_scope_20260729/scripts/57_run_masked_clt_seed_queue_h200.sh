#!/usr/bin/env bash
set -euo pipefail

# Run one prespecified BF16 seed across the three-model TopK CLT panel on one H200.
# Launch three copies with distinct GPU_ID/SEED pairs (17, 29, 43).

OSS_ROOT="${OSS_ROOT:-/oss-pvc/zhk_zip}"
REVISION_ROOT="${REVISION_ROOT:-${OSS_ROOT}/outputs/npj_revision_20260717}"
DICTIONARY_ROOT="${DICTIONARY_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/npj_revision_20260717/dictionaries_topk_bf16_r2}"
COHORT_ROOT="${COHORT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/npj_revision_20260717/cohorts}"
EXPECTED_PROJECT_ROOT="/gpfs/jiaotongdamoxing/zhk_zip/biocc/npj_revision_20260717/code_topk_confirmatory_bf16_r7/r2_interpretability_transfer"
EXPECTED_CODE_ARCHIVE_PATH="/gpfs/jiaotongdamoxing/zhk_zip/biocc/npj_revision_20260717/code_archives/r2_topk_confirmatory_bf16_payload_r7.tar"
EXPECTED_CODE_ARCHIVE_SHA256="${EXPECTED_CODE_ARCHIVE_SHA256:?set the SHA-256 of the new BF16 code archive}"
ARCHIVE_MEMBER_ROOT="r2_interpretability_transfer"
TREE_MANIFEST_NAME="CODE_CONTENT_SHA256SUMS"
PROJECT_ROOT="${PROJECT_ROOT:-${EXPECTED_PROJECT_ROOT}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/clt_training_confirmatory.yaml}"
CODE_ARCHIVE_PATH="${CODE_ARCHIVE_PATH:-${EXPECTED_CODE_ARCHIVE_PATH}}"
GPU_ID="${GPU_ID:?set GPU_ID to one allocated physical GPU index}"
SEED="${SEED:?set SEED to a prespecified independent seed}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
D_CLT="${D_CLT:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2}"
VERIFY_DEPLOYMENT_ONLY="${VERIFY_DEPLOYMENT_ONLY:-0}"
EXPECTED_CONFIG_SHA256="2f3ff5ae49553801a2ec07ac67a291716899253da04f3803accf8b09eee88adf"

export PYTHONDONTWRITEBYTECODE=1

case "${GPU_ID}" in
  0|1|2|3) ;;
  *) echo "GPU_ID must be one of the four allocated indices: 0,1,2,3" >&2; exit 2 ;;
esac
case "${SEED}" in
  17|29|43) ;;
  *) echo "SEED must be one of the frozen confirmatory seeds: 17,29,43" >&2; exit 2 ;;
esac
if [ "${TOTAL_STEPS}" -ne 200000 ] || [ "${D_CLT}" -ne 8192 ] || [ "${BATCH_SIZE}" -ne 2 ]; then
  echo "Frozen queue requires TOTAL_STEPS=200000, D_CLT=8192, BATCH_SIZE=2" >&2
  exit 2
fi
case "${VERIFY_DEPLOYMENT_ONLY}" in
  0|1) ;;
  *) echo "VERIFY_DEPLOYMENT_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
case "${DICTIONARY_ROOT}" in
  /gpfs/*) ;;
  *) echo "Confirmatory active checkpoints must be on GPFS" >&2; exit 2 ;;
esac
if [ "${PROJECT_ROOT}" != "${EXPECTED_PROJECT_ROOT}" ]; then
  echo "PROJECT_ROOT differs from the frozen deployment path" >&2
  exit 2
fi
if [ "${CODE_ARCHIVE_PATH}" != "${EXPECTED_CODE_ARCHIVE_PATH}" ]; then
  echo "CODE_ARCHIVE_PATH differs from the frozen deployment path" >&2
  exit 2
fi
test -f "${CODE_ARCHIVE_PATH}"
test -f "${PROJECT_ROOT}/${TREE_MANIFEST_NAME}"
ACTUAL_CONFIG_SHA256="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
if [ "${ACTUAL_CONFIG_SHA256}" != "${EXPECTED_CONFIG_SHA256}" ]; then
  echo "Confirmatory configuration SHA-256 mismatch" >&2
  exit 2
fi
ACTUAL_ARCHIVE_SHA256="$(sha256sum "${CODE_ARCHIVE_PATH}" | awk '{print $1}')"
if [ "${ACTUAL_ARCHIVE_SHA256}" != "${EXPECTED_CODE_ARCHIVE_SHA256}" ]; then
  echo "Code archive SHA-256 mismatch" >&2
  exit 2
fi
ARCHIVED_TREE_MANIFEST_SHA256="$(
  tar -xOf "${CODE_ARCHIVE_PATH}" \
    "${ARCHIVE_MEMBER_ROOT}/${TREE_MANIFEST_NAME}" | sha256sum | awk '{print $1}'
)"
DEPLOYED_TREE_MANIFEST_SHA256="$(
  sha256sum "${PROJECT_ROOT}/${TREE_MANIFEST_NAME}" | awk '{print $1}'
)"
if [ "${ARCHIVED_TREE_MANIFEST_SHA256}" != "${DEPLOYED_TREE_MANIFEST_SHA256}" ]; then
  echo "Deployed tree manifest is not the manifest inside the frozen archive" >&2
  exit 2
fi
if find "${PROJECT_ROOT}" -type l -print -quit | grep -q .; then
  echo "Deployed code tree must not contain symbolic links" >&2
  exit 2
fi
(
  cd "${PROJECT_ROOT}"
  sha256sum --check --strict "${TREE_MANIFEST_NAME}" >/dev/null
  manifest_inventory="$(mktemp)"
  deployed_inventory="$(mktemp)"
  trap 'rm -f "${manifest_inventory}" "${deployed_inventory}"' EXIT
  cut -c67- "${TREE_MANIFEST_NAME}" | LC_ALL=C sort > "${manifest_inventory}"
  find . -type f ! -name "${TREE_MANIFEST_NAME}" -print | LC_ALL=C sort \
    > "${deployed_inventory}"
  if ! cmp -s "${manifest_inventory}" "${deployed_inventory}"; then
    echo "Deployed code-tree inventory differs from the frozen archive manifest" >&2
    exit 2
  fi
)
LAUNCHER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
if [ "${VERIFY_DEPLOYMENT_ONLY}" = "1" ]; then
  echo "Verified frozen TopK deployment: archive_sha256=${EXPECTED_CODE_ARCHIVE_SHA256} deployed_tree_manifest_sha256=${DEPLOYED_TREE_MANIFEST_SHA256} launcher_sha256=${LAUNCHER_SHA256}"
  exit 0
fi

mkdir -p "${REVISION_ROOT}/logs"

run_one() {
  local model="$1"
  local cohort="$2"
  local fasta="$3"
  local input_format="$4"
  local expected_manifest_sha="$5"
  local manifest="${COHORT_ROOT}/${cohort}/train.jsonl"
  local manifest_sha
  local output_root="${DICTIONARY_ROOT}/${model}/topk_clt/seed_${SEED}"
  local final_manifest
  local final_manifest_sha
  local gpu_processes

  test -f "${manifest}"
  test -f "${fasta}"
  manifest_sha="$(sha256sum "${manifest}" | awk '{print $1}')"
  if [ "${manifest_sha}" != "${expected_manifest_sha}" ]; then
    echo "Frozen cohort SHA-256 mismatch: ${manifest}" >&2
    exit 2
  fi
  if [ -e "${output_root}" ]; then
    echo "Refusing to reuse any existing run directory: ${output_root}" >&2
    exit 2
  fi
  if ! gpu_processes="$(
    nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid \
      --format=csv,noheader,nounits
  )"; then
    echo "nvidia-smi process query failed for GPU ${GPU_ID}" >&2
    exit 2
  fi
  if grep -Eq '[0-9]' <<<"${gpu_processes}"; then
    echo "GPU ${GPU_ID} is occupied; refusing launch" >&2
    exit 2
  fi

  echo "[$(date --iso-8601=seconds)] preflight model=${model} seed=${SEED} gpu=${GPU_ID} model_inference_dtype=bfloat16 code_archive_sha256=${EXPECTED_CODE_ARCHIVE_SHA256} deployed_tree_manifest_sha256=${DEPLOYED_TREE_MANIFEST_SHA256} launcher_sha256=${LAUNCHER_SHA256} config_sha256=${ACTUAL_CONFIG_SHA256}"
  nvidia-smi --id="${GPU_ID}" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
  free -h

  PROJECT_ROOT="${PROJECT_ROOT}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  MODEL="${model}" \
  GPU_IDS="${GPU_ID}" \
  NUM_GPUS=1 \
  FASTA_PATH="${fasta}" \
  OUTPUT_ROOT="${output_root}" \
  D_CLT="${D_CLT}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TOTAL_STEPS="${TOTAL_STEPS}" \
  SEED="${SEED}" \
  CONFIRMATORY=1 \
  MANIFEST_PATH="${manifest}" \
  MANIFEST_SHA256="${manifest_sha}" \
  DATA_SPLIT=train \
  MODEL_INPUT_FORMAT="${input_format}" \
  bash "${PROJECT_ROOT}/scripts/04_run_h200_clt.sh"

  final_manifest="${output_root}/clt_weights/${model}/step_${TOTAL_STEPS}/checkpoint_manifest.json"
  test -f "${final_manifest}"
  final_manifest_sha="$(sha256sum "${final_manifest}" | awk '{print $1}')"
  echo "[$(date --iso-8601=seconds)] completed model=${model} seed=${SEED} gpu=${GPU_ID} final_checkpoint_manifest_sha256=${final_manifest_sha}"
  nvidia-smi --id="${GPU_ID}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  free -h
}

run_one protgpt2 uniref50 /gpfs/jiaotongdamoxing/zhk_zip/data/uniref50/uniref50.fasta sequence f4270143c1c22904d4f548de9837572040d2b10b87d9c64363fdb811422525bb
run_one zymctrl zymctrl /gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/ec_labeled.fasta zymctrl_ec f9727a374209f7329bed910624430fcf747b735314895215d1eff313b0d3f755
run_one progen2-medium uniref50 /gpfs/jiaotongdamoxing/zhk_zip/data/uniref50/uniref50.fasta sequence f4270143c1c22904d4f548de9837572040d2b10b87d9c64363fdb811422525bb

echo "[$(date --iso-8601=seconds)] seed queue complete seed=${SEED} gpu=${GPU_ID}"
