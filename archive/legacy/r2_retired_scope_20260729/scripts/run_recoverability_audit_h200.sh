#!/usr/bin/env bash
set -uo pipefail

# Run the R2 recoverability audit on an H200 pod. This orchestrates cache ->
# probes -> oracle steering -> decision, then launches the gated retrain
# (script 48) only if the frozen decision table returns GO.

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CODE_ROOT="${CODE_ROOT:-/oss-pvc/zhk_zip/biocc/Research2}"
RESULT_ROOT="${RESULT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2}"
OUT_ROOT="${OUT_ROOT:-${RESULT_ROOT}/results/representation_audit_${STAMP}}"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-/gpfs/jiaotongdamoxing/zhk_zip/models}"

GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export R2_MODEL_BASE_DIR="${MODEL_BASE_DIR}"
export PYTHONUNBUFFERED=1

SWISSPROT_CACHE="${SWISSPROT_CACHE:-/gpfs/jiaotongdamoxing/zhk_zip/data/processed/swissprot_all_max1022.pkl}"
PFAM_RESIDUE="${PFAM_RESIDUE:-/gpfs/jiaotongdamoxing/zhk_zip/data/interpro/pfam_residue.tsv}"
EC_FASTA="${EC_FASTA:-/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/ec_labeled_swissprot.fasta}"
DECODER_EC_JSON="${DECODER_EC_JSON:-/oss-pvc/zhk_zip/biocc/Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json}"
ESM2_MODEL="${ESM2_MODEL:-/gpfs/jiaotongdamoxing/zhk_zip/models/esm2_t36_3B_UR50D}"

PROTGPT2_CLT="${PROTGPT2_CLT:-/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000}"
ZYMCTRL_CLT="${ZYMCTRL_CLT:-/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000}"
PROGEN2_CLT="${PROGEN2_CLT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000}"

LAYERS="${LAYERS:-all}"
RESIDUE_LAYERS="${RESIDUE_LAYERS:-even6}"
ESM_BATCH_SIZE="${ESM_BATCH_SIZE:-1}"
LIMIT_SEQUENCES="${LIMIT_SEQUENCES:-0}"
TASKS="${TASKS:-ec_topclass,pfam_family,secondary_fraction,residue_ss,decoder_ec}"
N_BOOT="${N_BOOT:-1000}"
RUN_46="${RUN_46:-1}"
N_GEN_46="${N_GEN_46:-40}"
RUN_48="${RUN_48:-1}"
RETRAIN_TOTAL_STEPS="${RETRAIN_TOTAL_STEPS:-300000}"
RETRAIN_WIDTH_FACTOR="${RETRAIN_WIDTH_FACTOR:-4}"
RETRAIN_BATCH_SIZE="${RETRAIN_BATCH_SIZE:-2}"
RETRAIN_NUM_SEQUENCES="${RETRAIN_NUM_SEQUENCES:-300000}"
RETRAIN_OUT_NAME="${RETRAIN_OUT_NAME:-r2_clt_recoverability_retrain_${STAMP}}"
RETRAIN_FASTA_PATH="${RETRAIN_FASTA_PATH:-/gpfs/jiaotongdamoxing/zhk_zip/data/uniref50/uniref50.fasta}"

CACHE_DIR="${OUT_ROOT}/cache"
PROBES_DIR="${OUT_ROOT}/probes"
STEERING_DIR="${OUT_ROOT}/steering"
DECISION_DIR="${OUT_ROOT}/decision"
LOG_DIR="${RESULT_ROOT}/logs/runtime"
mkdir -p "${CACHE_DIR}" "${PROBES_DIR}" "${STEERING_DIR}" "${DECISION_DIR}" "${LOG_DIR}"

cd "${CODE_ROOT}" || exit 2

echo "[run] stamp=${STAMP}"
echo "[run] code_root=${CODE_ROOT}"
echo "[run] out_root=${OUT_ROOT}"
echo "[run] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[run] model_base_dir=${MODEL_BASE_DIR}"
nvidia-smi || true

limit_args=()
if [[ "${LIMIT_SEQUENCES}" != "0" ]]; then
  limit_args=(--limit-sequences "${LIMIT_SEQUENCES}")
fi

echo "[run] 44 cache"
python3 scripts/44_cache_representations.py \
  --model-spec "protgpt2=${PROTGPT2_CLT}" \
  --model-spec "zymctrl=${ZYMCTRL_CLT}" \
  --model-spec "progen2-medium=${PROGEN2_CLT}" \
  --model-base-dir "${MODEL_BASE_DIR}" \
  --esm2-model "${ESM2_MODEL}" \
  --swissprot-cache "${SWISSPROT_CACHE}" \
  --pfam-residue "${PFAM_RESIDUE}" \
  --ec-fasta "${EC_FASTA}" \
  --decoder-ec-json "${DECODER_EC_JSON}" \
  --layers "${LAYERS}" \
  --residue-layers "${RESIDUE_LAYERS}" \
  --esm-batch-size "${ESM_BATCH_SIZE}" \
  --device cuda \
  --out-dir "${CACHE_DIR}" \
  "${limit_args[@]}"
status44=$?
echo "[run] 44 status=${status44}"
if [[ "${status44}" != "0" ]]; then
  exit "${status44}"
fi

echo "[run] 45 probes"
python3 scripts/45_probe_ceiling_floor.py \
  --cache-dir "${CACHE_DIR}" \
  --out-dir "${PROBES_DIR}" \
  --tasks "${TASKS}" \
  --n-boot "${N_BOOT}"
status45=$?
echo "[run] 45 status=${status45}"
if [[ "${status45}" != "0" ]]; then
  exit "${status45}"
fi

status46=0
if [[ "${RUN_46}" == "1" ]]; then
  echo "[run] 46 oracle steering"
  python3 scripts/46_oracle_direction_steering.py \
    --cache-dir "${CACHE_DIR}" \
    --out-dir "${STEERING_DIR}" \
    --model-base-dir "${MODEL_BASE_DIR}" \
    --model zymctrl \
    --inject-layer 3 \
    --n-gen "${N_GEN_46}" \
    --device cuda
  status46=$?
  echo "[run] 46 status=${status46}"
fi

echo "[run] 47 decision"
python3 scripts/47_decision_table.py \
  --probes "${PROBES_DIR}/probe_results.json" \
  --steering "${STEERING_DIR}/oracle_steering.json" \
  --out-dir "${DECISION_DIR}"
status47=$?
echo "[run] 47 status=${status47}"
if [[ "${status47}" != "0" ]]; then
  exit "${status47}"
fi

decision=$(python3 -c "import json; d=json.load(open('${DECISION_DIR}/decision.json'))['retrain_decision']; print(d.get('decision',''))")
target=$(python3 -c "import json; d=json.load(open('${DECISION_DIR}/decision.json'))['retrain_decision']; print(d.get('retrain_target') or '')")

status48=0
status44r=0
status45r=0
if [[ "${RUN_48}" == "1" && "${decision}" == "GO" && -n "${target}" ]]; then
  echo "[run] 48 gated retrain target=${target}"
  python3 scripts/48_capacity_retrain.py \
    --decision "${DECISION_DIR}/decision.json" \
    --target "${target}" \
    --gpu "${GPU}" \
    --out-name "${RETRAIN_OUT_NAME}" \
    --width-factor "${RETRAIN_WIDTH_FACTOR}" \
    --total-steps "${RETRAIN_TOTAL_STEPS}" \
    --batch-size "${RETRAIN_BATCH_SIZE}" \
    --num-sequences "${RETRAIN_NUM_SEQUENCES}" \
    --fasta-path "${RETRAIN_FASTA_PATH}" \
    --execute
  status48=$?
  echo "[run] 48 status=${status48}"
  if [[ "${status48}" != "0" ]]; then
    exit "${status48}"
  fi

  retrain_ckpt="/oss-pvc/zhk_zip/outputs/${RETRAIN_OUT_NAME}/clt_weights/${target}/step_${RETRAIN_TOTAL_STEPS}"
  if [[ ! -f "${retrain_ckpt}/clt.pt" ]]; then
    echo "[run] missing retrain checkpoint: ${retrain_ckpt}/clt.pt"
    exit 3
  fi

  echo "[run] 44 retrain cache target=${target}"
  python3 scripts/44_cache_representations.py \
    --model-spec "${target}=${retrain_ckpt}" \
    --model-base-dir "${MODEL_BASE_DIR}" \
    --esm2-model "${ESM2_MODEL}" \
    --swissprot-cache "${SWISSPROT_CACHE}" \
    --pfam-residue "${PFAM_RESIDUE}" \
    --ec-fasta "${EC_FASTA}" \
    --decoder-ec-json "${DECODER_EC_JSON}" \
    --layers "${LAYERS}" \
    --residue-layers "${RESIDUE_LAYERS}" \
    --esm-batch-size "${ESM_BATCH_SIZE}" \
    --device cuda \
    --out-dir "${OUT_ROOT}/cache_retrain" \
    "${limit_args[@]}"
  status44r=$?
  echo "[run] 44 retrain status=${status44r}"
  if [[ "${status44r}" != "0" ]]; then
    exit "${status44r}"
  fi

  echo "[run] 45 retrain probes"
  python3 scripts/45_probe_ceiling_floor.py \
    --cache-dir "${OUT_ROOT}/cache_retrain" \
    --out-dir "${OUT_ROOT}/probes_retrain" \
    --tasks "${TASKS}" \
    --n-boot "${N_BOOT}"
  status45r=$?
  echo "[run] 45 retrain status=${status45r}"
  if [[ "${status45r}" != "0" ]]; then
    exit "${status45r}"
  fi
else
  echo "[run] 48 skipped: RUN_48=${RUN_48} decision=${decision} target=${target}"
fi

echo "[run] done statuses: 44=${status44} 45=${status45} 46=${status46} 47=${status47} 48=${status48} 44r=${status44r} 45r=${status45r}"
nvidia-smi || true
exit 0
