#!/usr/bin/env bash
set -euo pipefail

# Submit an H200 pod for Research1 SAE training.
#
# Example:
#   bash r1_encoder_interpretability_benchmark/scripts/submit_h200_sae.sh
#   JOB_NAME=jiaotongdamoxing-zhk_lzp-r1-sae GPUS=4 bash r1_encoder_interpretability_benchmark/scripts/submit_h200_sae.sh

JOB_PREFIX="${JOB_PREFIX:-jiaotongdamoxing-zhk_lzp}"
JOB_NAME="${JOB_NAME:-${JOB_PREFIX}-r1-sae-$(date +%m%d-%H%M)}"
IMAGE="${IMAGE:-cr-ee.registry.cn-hangzhou-cicore-d01.res.cncicore.com/bmcp-private/vllm:0.9.0.1-pytorch2.7-cu128-20250612}"
GPUS="${GPUS:-2}"
PVC="${PVC:-damoxing}"
OSS_ROOT="${OSS_ROOT:-/oss-pvc/zhk_zip}"
PROJECT_ROOT="${PROJECT_ROOT:-${OSS_ROOT}/biocc/Research1}"

echo "Submitting Research1 H200 pod..."
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
