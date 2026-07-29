#!/usr/bin/env bash
set -uo pipefail

# Run the R2 v2 remaining queue, then keep the pod alive for follow-up
# work instead of letting Arena release the GPUs when the script exits.

STAMP="${STAMP:-20260424_r2_v2_remaining}"
N_BENCH="${N_BENCH:-200}"
N_CASE="${N_CASE:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
STRUCTURAL_CUDA_DEVICE="${STRUCTURAL_CUDA_DEVICE:-0}"
STRUCTURAL_DTYPE="${STRUCTURAL_DTYPE:-fp32}"
RESULT_ROOT="${RESULT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi}"
LOG_DIR="${RESULT_ROOT}/logs/runtime"
mkdir -p "${LOG_DIR}"

echo "[$(date '+%F %T')] starting R2 v2 remaining queue, then hold"
STAMP="${STAMP}" \
N_BENCH="${N_BENCH}" \
N_CASE="${N_CASE}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
STRUCTURAL_CUDA_DEVICE="${STRUCTURAL_CUDA_DEVICE}" \
STRUCTURAL_DTYPE="${STRUCTURAL_DTYPE}" \
bash scripts/run_r2_v2_remaining_0424.sh
status=$?

echo "[$(date '+%F %T')] queue exited with status ${status}; holding pod"
printf '%s\n' "${status}" > "${LOG_DIR}/r2_v2_remaining_then_hold_exit_status_${STAMP}.txt"

tail -f /dev/null
