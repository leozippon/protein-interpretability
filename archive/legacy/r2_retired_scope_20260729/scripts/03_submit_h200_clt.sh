#!/usr/bin/env bash
set -euo pipefail

# Submit a 1-GPU H200 pod for Research2 CLT training.
#
# Example:
#   bash r2_interpretability_transfer/scripts/03_submit_h200_clt.sh
#   JOB_NAME=jiaotongdamoxing-zhk_lzp-r2-protgpt2 bash r2_interpretability_transfer/scripts/03_submit_h200_clt.sh

JOB_PREFIX="${JOB_PREFIX:-jiaotongdamoxing-zhk_lzp}"
JOB_NAME="${JOB_NAME:-${JOB_PREFIX}-r2-clt-$(date +%m%d-%H%M)}"
IMAGE="${IMAGE:-cr-ee.registry.cn-hangzhou-cicore-d01.res.cncicore.com/bmcp-private/vllm:0.9.0.1-pytorch2.7-cu128-20250612}"
GPUS="${GPUS:-1}"
PVC="${PVC:-damoxing}"
OSS_ROOT="${OSS_ROOT:-/oss-pvc/zhk_zip}"
PROJECT_ROOT="${PROJECT_ROOT:-${OSS_ROOT}/biocc/paper_r2_nature_mi}"

echo "Submitting Research2 H200 pod..."
echo "  JOB_NAME:     ${JOB_NAME}"
echo "  JOB_PREFIX:   ${JOB_PREFIX}"
echo "  IMAGE:        ${IMAGE}"
echo "  GPUS:         ${GPUS}"
echo "  PVC:          ${PVC}"
echo "  PROJECT_ROOT: ${PROJECT_ROOT}"

arena submit pytorch \
  --name="${JOB_NAME}" \
  --gpus="${GPUS}" \
  --image="${IMAGE}" \
  --data="${PVC}:/oss-pvc" \
  --data-dir=/gpfs \
  --working-dir="${PROJECT_ROOT}" \
  --toleration=all \
  -- \
  'tail -f /dev/null'

echo
echo "Monitor with:"
echo "  kubectl get pods | grep ${JOB_NAME}"
echo "  kubectl exec -it ${JOB_NAME}-master-0 -- bash"
