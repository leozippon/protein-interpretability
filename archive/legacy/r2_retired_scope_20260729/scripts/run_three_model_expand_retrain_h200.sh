#!/usr/bin/env bash
set -uo pipefail

# Exploratory override: train expanded CLT dictionaries for all three R2
# decoder models. This is intentionally outside the frozen recoverability
# NO-GO gate and should be reported as exploratory.
#
# With the current 2-GPU beliefnav pod, each model is trained with 2-GPU DDP
# and the three model jobs run sequentially. Running all three 2-GPU jobs at
# once would require six GPUs.

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CODE_ROOT="${CODE_ROOT:-/oss-pvc/zhk_zip/biocc/Research2}"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-/gpfs/jiaotongdamoxing/zhk_zip/models}"
OUT_NAME="${OUT_NAME:-r2_clt_expand_retrain_${STAMP}}"
SAVE_ROOT="${SAVE_ROOT:-/oss-pvc/zhk_zip/outputs/research2/clt_weights/${OUT_NAME}}"
LOG_ROOT="${LOG_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/${OUT_NAME}}"

GPUS="${GPUS:-0,1}"
TOTAL_STEPS="${TOTAL_STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SEQUENCES="${NUM_SEQUENCES:-300000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50000}"
LR="${LR:-0.0003}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-3000}"
D_CLT_PROTGPT2="${D_CLT_PROTGPT2:-16384}"
D_CLT_ZYMCTRL="${D_CLT_ZYMCTRL:-16384}"
D_CLT_PROGEN2="${D_CLT_PROGEN2:-16384}"
K_PROTGPT2="${K_PROTGPT2:-128}"
K_ZYMCTRL="${K_ZYMCTRL:-128}"
K_PROGEN2="${K_PROGEN2:-128}"
WINDOW="${WINDOW:-8}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-2500}"
DEAD_THRESHOLD="${DEAD_THRESHOLD:-5000}"
FORCE="${FORCE:-0}"

UNIREF_FASTA="${UNIREF_FASTA:-/gpfs/jiaotongdamoxing/zhk_zip/data/uniref50/uniref50.fasta}"
ZYMCTRL_FASTA="${ZYMCTRL_FASTA:-/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/ec_labeled.fasta}"

export R2_MODEL_BASE_DIR="${MODEL_BASE_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${LOG_ROOT}" "${SAVE_ROOT}"
cd "${CODE_ROOT}" || exit 2

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

latest_step_dir() {
  local save_dir="$1"
  local best_step=-1
  local best_dir=""
  local d base step
  shopt -s nullglob
  for d in "${save_dir}"/step_*; do
    base="${d##*/}"
    step="${base#step_}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step > best_step )); then
      best_step="${step}"
      best_dir="${d}"
    fi
  done
  shopt -u nullglob
  if (( best_step >= 0 )); then
    printf '%s\n' "${best_dir}"
  fi
}

run_model() {
  local model="$1"
  local config="$2"
  local fasta="$3"
  local d_clt="$4"
  local k="$5"
  local save_dir="${SAVE_ROOT}/${model}"
  local final_dir="${save_dir}/step_${TOTAL_STEPS}"
  local log_file="${LOG_ROOT}/${model}.log"

  if [[ "${FORCE}" != "1" && -f "${final_dir}/clt.pt" ]]; then
    log "SKIP ${model}: final checkpoint exists at ${final_dir}"
    return 0
  fi

  local resume_args=()
  local latest
  latest="$(latest_step_dir "${save_dir}")"
  if [[ -n "${latest}" && "${FORCE}" != "1" ]]; then
    resume_args=(--resume "${latest}")
    log "RESUME ${model}: ${latest}"
  fi

  log "START ${model}"
  log "CONFIG ${config}"
  log "SAVE ${save_dir}"
  log "LOG ${log_file}"
  log "DDP_GPUS ${GPUS} d_clt=${d_clt} k=${k} batch=${BATCH_SIZE}/gpu steps=${TOTAL_STEPS}"

  python3 scripts/01_train_clt.py \
    --config "${config}" \
    --gpus "${GPUS}" \
    "${resume_args[@]}" \
    --override \
      "model.name=${model}" \
      "model.dtype=float16" \
      "data.fasta_path=${fasta}" \
      "data.max_seq_len=${MAX_SEQ_LEN}" \
      "data.num_sequences=${NUM_SEQUENCES}" \
      "clt.d_clt=${d_clt}" \
      "clt.k=${k}" \
      "clt.window=${WINDOW}" \
      "clt.resample_every=${RESAMPLE_EVERY}" \
      "clt.dead_feature_threshold=${DEAD_THRESHOLD}" \
      "training.batch_size=${BATCH_SIZE}" \
      "training.total_steps=${TOTAL_STEPS}" \
      "training.lr=${LR}" \
      "training.lr_warmup_steps=${LR_WARMUP_STEPS}" \
      "checkpoint.save_dir=${save_dir}" \
      "checkpoint.save_every_steps=${SAVE_EVERY_STEPS}" \
      "logging.wandb_project=null" \
      "logging.log_every_steps=50" \
    2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  log "DONE ${model} status=${status}"
  return "${status}"
}

log "Three-model expanded CLT retrain"
log "STAMP=${STAMP}"
log "CODE_ROOT=${CODE_ROOT}"
log "SAVE_ROOT=${SAVE_ROOT}"
log "LOG_ROOT=${LOG_ROOT}"
log "MODEL_BASE_DIR=${MODEL_BASE_DIR}"
nvidia-smi || true

for required in "${UNIREF_FASTA}" "${ZYMCTRL_FASTA}" \
  "${MODEL_BASE_DIR}/ProtGPT2" "${MODEL_BASE_DIR}/ZymCTRL" "${MODEL_BASE_DIR}/progen2-medium"; do
  if [[ ! -e "${required}" ]]; then
    log "MISSING required path: ${required}"
    exit 2
  fi
done

run_model protgpt2 configs/clt_training_protgpt2_v2.yaml "${UNIREF_FASTA}" "${D_CLT_PROTGPT2}" "${K_PROTGPT2}" || exit $?
run_model zymctrl configs/clt_training_zymctrl_v2.yaml "${ZYMCTRL_FASTA}" "${D_CLT_ZYMCTRL}" "${K_ZYMCTRL}" || exit $?
run_model progen2-medium configs/clt_training.yaml "${UNIREF_FASTA}" "${D_CLT_PROGEN2}" "${K_PROGEN2}" || exit $?

log "ALL DONE"
nvidia-smi || true
