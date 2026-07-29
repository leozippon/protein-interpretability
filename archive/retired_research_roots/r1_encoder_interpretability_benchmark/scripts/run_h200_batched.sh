#!/usr/bin/env bash
# Run Research1 SAE training in batched mode on H200.
#
# Default behavior:
# - Uses the GPUs listed in GPU_IDS (default: 0,1,2,3)
# - Reads code from OSS mount under /oss-pvc
# - Reads model + FASTA from GPFS
# - Writes logs/checkpoints back to OSS
# - Disables wandb by default for offline H200 pods
# - Canonical remote roots are /oss-pvc/zhk_zip and /gpfs/jiaotongdamoxing/zhk_zip
#
# Typical usage inside a prepared H200 pod:
#   bash scripts/run_h200_batched.sh
#
# Optional overrides:
#   USER_DIR=zhk_zip \
#   LAB_DIR=labspace \
#   GPU_IDS=0,1,2,3 \
#   LAYERS=19,23,27,31,35 \
#   OUTPUT_NAME=r1_batch_20260401 \
#   VENV_PATH=/oss-pvc/zhk_zip/venvs/r1 \
#   bash scripts/run_h200_batched.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/sae_training.yaml}"

LAB_DIR="${LAB_DIR:-jiaotongdamoxing}"
USER_DIR="${USER_DIR:-zhk_zip}"
OSS_ROOT="${OSS_ROOT:-/oss-pvc/${USER_DIR}}"
GPFS_ROOT="${GPFS_ROOT:-/gpfs/${LAB_DIR}/${USER_DIR}}"

MODEL_DIR="${MODEL_DIR:-${GPFS_ROOT}/models/esm2_t36_3B_UR50D}"
FASTA_PATH="${FASTA_PATH:-${GPFS_ROOT}/data/uniref50/uniref50.fasta}"

OUTPUT_NAME="${OUTPUT_NAME:-r1_h200_batched}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OSS_ROOT}/outputs/${OUTPUT_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/sae_weights}"

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

if [[ -n "${VENV_PATH:-}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: config not found at ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
    echo "ERROR: project src/ directory not found under ${PROJECT_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: model directory missing config.json at ${MODEL_DIR}" >&2
    exit 1
fi

if [[ ! -f "${FASTA_PATH}" ]]; then
    echo "ERROR: FASTA file not found at ${FASTA_PATH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

if [[ -n "${LAYERS:-}" ]]; then
    IFS=',' read -r -a TARGET_LAYERS <<< "${LAYERS}"
else
    mapfile -t TARGET_LAYERS < <(
        python3 - <<'PY' "${CONFIG_PATH}"
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

for layer in cfg["training"]["layers_to_train"]:
    print(layer)
PY
    )
fi

if [[ "${#TARGET_LAYERS[@]}" -eq 0 ]]; then
    echo "ERROR: no target layers resolved" >&2
    exit 1
fi

echo "============================================"
echo "Research1 H200 Batched Launcher"
echo "============================================"
echo "PROJECT_ROOT:   ${PROJECT_ROOT}"
echo "CONFIG_PATH:    ${CONFIG_PATH}"
echo "OSS_ROOT:       ${OSS_ROOT}"
echo "GPFS_ROOT:      ${GPFS_ROOT}"
echo "MODEL_DIR:      ${MODEL_DIR}"
echo "FASTA_PATH:     ${FASTA_PATH}"
echo "OUTPUT_ROOT:    ${OUTPUT_ROOT}"
echo "GPU_IDS:        ${GPU_IDS[*]}"
echo "TARGET_LAYERS:  ${TARGET_LAYERS[*]}"
echo ""

failed=0
batch_size="${#GPU_IDS[@]}"

for ((batch_start=0; batch_start<${#TARGET_LAYERS[@]}; batch_start+=batch_size)); do
    batch_end=$((batch_start + batch_size))
    if (( batch_end > ${#TARGET_LAYERS[@]} )); then
        batch_end=${#TARGET_LAYERS[@]}
    fi

    echo "----- Launching batch ${batch_start}-${batch_end} -----"

    pids=()
    batch_layers=()

    for ((i=batch_start; i<batch_end; i++)); do
        gpu_index=$((i - batch_start))
        gpu="${GPU_IDS[$gpu_index]}"
        layer="${TARGET_LAYERS[$i]}"
        logfile="${LOG_DIR}/train_layer${layer}.log"

        batch_layers+=("${layer}")
        echo "Layer ${layer} -> GPU ${gpu}, log=${logfile}"

        env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu}" \
            python3 "${PROJECT_ROOT}/scripts/02_train_saes.py" \
                --config "${CONFIG_PATH}" \
                --layer "${layer}" \
                --gpus 0 \
                --override \
                    "model.name_or_path=${MODEL_DIR}" \
                    "data.fasta_path=${FASTA_PATH}" \
                    "checkpoint.save_dir=${CHECKPOINT_DIR}" \
                    "logging.wandb_project=null" \
            > "${logfile}" 2>&1 &

        pids+=($!)
    done

    for ((j=0; j<${#pids[@]}; j++)); do
        pid="${pids[$j]}"
        layer="${batch_layers[$j]}"
        if wait "${pid}"; then
            echo "Layer ${layer}: DONE"
        else
            echo "Layer ${layer}: FAILED" >&2
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "============================================"
echo "Training batches complete"
echo "Logs:        ${LOG_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "============================================"

if (( failed > 0 )); then
    echo "ERROR: ${failed} layer job(s) failed" >&2
    exit 1
fi
