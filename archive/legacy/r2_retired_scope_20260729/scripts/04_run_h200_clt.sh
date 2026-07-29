#!/usr/bin/env bash
set -euo pipefail

# Run Research2 CLT training inside an H200 pod with offline model/data paths.
# Canonical remote roots are /oss-pvc/zhk_zip and /gpfs/jiaotongdamoxing/zhk_zip.
#
# Single GPU (default):
#   MODEL=protgpt2 OUTPUT_NAME=r2_clt_protgpt2_test bash scripts/04_run_h200_clt.sh
#
# Multi-GPU DDP (2 GPUs):
#   NUM_GPUS=2 MODEL=protgpt2 OUTPUT_NAME=r2_clt_protgpt2_ddp bash scripts/04_run_h200_clt.sh

PROJECT_ROOT="${PROJECT_ROOT:-/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/clt_training_h200.yaml}"
OSS_ROOT="${OSS_ROOT:-/oss-pvc/zhk_zip}"
GPFS_ROOT="${GPFS_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip}"

MODEL="${MODEL:-protgpt2}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS:-}"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-${GPFS_ROOT}/models}"
FASTA_PATH="${FASTA_PATH:-${GPFS_ROOT}/data/uniref50/uniref50.fasta}"
OUTPUT_NAME="${OUTPUT_NAME:-r2_clt_${MODEL}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OSS_ROOT}/outputs/${OUTPUT_NAME}}"
D_CLT="${D_CLT:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
RESUME="${RESUME:-}"
SEED="${SEED:-0}"
CONFIRMATORY="${CONFIRMATORY:-0}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
MANIFEST_SHA256="${MANIFEST_SHA256:-}"
DATA_SPLIT="${DATA_SPLIT:-train}"
MODEL_INPUT_FORMAT="${MODEL_INPUT_FORMAT:-sequence}"
NUM_SEQUENCES="${NUM_SEQUENCES:-}"

if [ "${CONFIRMATORY}" = "1" ] && [ -z "${MANIFEST_PATH}" ]; then
  echo "CONFIRMATORY=1 requires MANIFEST_PATH" >&2
  exit 2
fi
case "${CONFIRMATORY}" in
  0|1) ;;
  *) echo "CONFIRMATORY must be 0 or 1" >&2; exit 2 ;;
esac
export CONFIRMATORY

if [ -n "${MANIFEST_PATH}" ]; then
  if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "Missing cohort manifest: ${MANIFEST_PATH}" >&2
    exit 2
  fi
  ACTUAL_MANIFEST_SHA256="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
  if [ -n "${MANIFEST_SHA256}" ] && [ "${ACTUAL_MANIFEST_SHA256}" != "${MANIFEST_SHA256}" ]; then
    echo "Manifest SHA-256 mismatch" >&2
    exit 2
  fi
  MANIFEST_SHA256="${ACTUAL_MANIFEST_SHA256}"
  if [ -z "${NUM_SEQUENCES}" ]; then
    NUM_SEQUENCES="$(wc -l < "${MANIFEST_PATH}")"
  fi
fi
if [ -z "${RESUME}" ] && [ -e "${OUTPUT_ROOT}" ]; then
  echo "Refusing to reuse fresh-run output: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if [ -n "${RESUME}" ] && [ ! -d "${OUTPUT_ROOT}" ]; then
  echo "Resume output directory does not exist: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/clt_weights"

export R2_MODEL_BASE_DIR="${MODEL_BASE_DIR}"
export R2_FASTA_PATH="${FASTA_PATH}"

echo "============================================"
echo "Research2 H200 CLT Launcher"
echo "============================================"
echo "PROJECT_ROOT:   ${PROJECT_ROOT}"
echo "CONFIG_PATH:    ${CONFIG_PATH}"
echo "MODEL:          ${MODEL}"
echo "MODEL_BASE_DIR: ${MODEL_BASE_DIR}"
echo "FASTA_PATH:     ${FASTA_PATH}"
echo "OUTPUT_ROOT:    ${OUTPUT_ROOT}"
echo "NUM_GPUS:       ${NUM_GPUS}"
echo "GPU_IDS:        ${GPU_IDS:-auto}"
echo "D_CLT:          ${D_CLT}"
echo "BATCH_SIZE:     ${BATCH_SIZE}"
echo "TOTAL_STEPS:    ${TOTAL_STEPS}"
echo "RESUME:         ${RESUME:-none}"
echo "SEED:           ${SEED}"
echo "CONFIRMATORY:   ${CONFIRMATORY}"
echo "MANIFEST_PATH:  ${MANIFEST_PATH:-none}"
echo "MANIFEST_SHA256:${MANIFEST_SHA256:-none}"
echo "DATA_SPLIT:     ${DATA_SPLIT}"
echo "INPUT_FORMAT:   ${MODEL_INPUT_FORMAT}"
echo "NUM_SEQUENCES:  ${NUM_SEQUENCES:-config default}"

LOGFILE="${OUTPUT_ROOT}/logs/clt_${MODEL}.log"

# Build or validate the physical GPU list.
if [ -n "${GPU_IDS}" ]; then
  if ! [[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must be a comma-separated list of integers" >&2
    exit 2
  fi
  GPU_LIST="${GPU_IDS}"
  GPU_COUNT="$(awk -F, '{print NF}' <<< "${GPU_LIST}")"
  if [ "${GPU_COUNT}" -ne "${NUM_GPUS}" ]; then
    echo "GPU_IDS count (${GPU_COUNT}) does not equal NUM_GPUS (${NUM_GPUS})" >&2
    exit 2
  fi
else
  GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

OVERRIDES=(
  "model.name=${MODEL}"
  "data.fasta_path=${FASTA_PATH}"
  "checkpoint.save_dir=${OUTPUT_ROOT}/clt_weights/${MODEL}"
  "logging.wandb_project=null"
  "clt.d_clt=${D_CLT}"
  "training.batch_size=${BATCH_SIZE}"
  "training.total_steps=${TOTAL_STEPS}"
  "training.seed=${SEED}"
)
if [ "${CONFIRMATORY}" = "1" ]; then
  OVERRIDES+=(
    "model.inference_dtype=bfloat16"
    "model.inference_dtype_verification=all_floating_model_parameters_exactly_declared_before_first_activation"
  )
fi
if [ -n "${MANIFEST_PATH}" ]; then
  OVERRIDES+=(
    "data.manifest_path=${MANIFEST_PATH}"
    "data.manifest_sha256=${MANIFEST_SHA256}"
    "data.split=${DATA_SPLIT}"
    "data.model_input_format=${MODEL_INPUT_FORMAT}"
    "data.num_sequences=${NUM_SEQUENCES}"
  )
fi

TRAIN_ARGS=(
  "${PROJECT_ROOT}/scripts/01_train_clt.py"
  --config "${CONFIG_PATH}"
  --gpus "${GPU_LIST}"
)
if [ -n "${RESUME}" ]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi
TRAIN_ARGS+=(--override "${OVERRIDES[@]}")

if [ -n "${RESUME}" ]; then
  TEE_ARGS=(-a "${LOGFILE}")
else
  TEE_ARGS=("${LOGFILE}")
fi
PYTHONUNBUFFERED=1 \
python3 "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${TEE_ARGS[@]}"

FINAL_CHECKPOINT="${OUTPUT_ROOT}/clt_weights/${MODEL}/step_${TOTAL_STEPS}"
python3 "${PROJECT_ROOT}/scripts/01_train_clt.py" \
  --config "${CONFIG_PATH}" \
  --gpus "${GPU_LIST}" \
  --verify-checkpoint "${FINAL_CHECKPOINT}" \
  --expected-step "${TOTAL_STEPS}" \
  --override "${OVERRIDES[@]}"
