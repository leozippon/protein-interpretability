#!/usr/bin/env bash
set -euo pipefail

# Finish the remaining R2 v2 post-training TODOs on an H200 pod.
# This script is meant to run inside an Arena pod with /oss-pvc and /gpfs
# mounted. It is intentionally restart-friendly: existing outputs are reused
# unless FORCE=1 is set.

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CODE_ROOT="${CODE_ROOT:-/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi}"
RESULT_ROOT="${RESULT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi}"
MODEL_ROOT="${MODEL_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/models}"
ZYMCTRL_V2_CKPT="${ZYMCTRL_V2_CKPT:-/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000}"
PROGEN2_CKPT="${PROGEN2_CKPT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000}"
EC_FEATURES="${EC_FEATURES:-${RESULT_ROOT}/results/circuit_analysis/zymctrl/ec_features.pkl}"
LYSOZYME_LEADS="${LYSOZYME_LEADS:-${RESULT_ROOT}/results/drug_design/ec_lysozyme_leads_v2.json}"
LYSOZYME_FASTA="${LYSOZYME_FASTA:-/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/data/ec_reference/ec_lysozyme_top10.fasta}"
ESMFOLD_PATH="${ESMFOLD_PATH:-/gpfs/jiaotongdamoxing/zhk_zip/models/esmfold_v1}"

N_BENCH="${N_BENCH:-200}"
N_CASE="${N_CASE:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
FORCE="${FORCE:-0}"
STRUCTURAL_CUDA_DEVICE="${STRUCTURAL_CUDA_DEVICE:-0}"
STRUCTURAL_DTYPE="${STRUCTURAL_DTYPE:-fp32}"

LOG_DIR="${RESULT_ROOT}/logs/runtime"
mkdir -p "${LOG_DIR}" \
  "${RESULT_ROOT}/results/steering_benchmark" \
  "${RESULT_ROOT}/results/drug_design" \
  "${RESULT_ROOT}/results/causal_ablation" \
  "${RESULT_ROOT}/results/circuit_analysis"

export PYTHONUNBUFFERED=1
export R2_MODEL_BASE_DIR="${MODEL_ROOT}"
export ESMFOLD_PATH

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_logged() {
  local name="$1"
  shift
  local logfile="${LOG_DIR}/${name}_${STAMP}.log"
  log "START ${name}"
  log "LOG ${logfile}"
  bash -lc "$*" 2>&1 | tee "${logfile}"
  log "DONE ${name}"
}

try_logged() {
  local name="$1"
  shift
  if ! run_logged "$name" "$@"; then
    log "FAILED ${name}; continuing"
    return 0
  fi
}

skip_or_run() {
  local output="$1"
  shift
  if [[ "${FORCE}" != "1" && -s "${output}" ]]; then
    log "SKIP existing output: ${output}"
    return 0
  fi
  "$@"
}

prepare_runtime() {
  mkdir -p /Data/public
  ln -sfn "${MODEL_ROOT}" /Data/public/models_R2
  if [[ -d "${ESMFOLD_PATH}" ]]; then
    ln -sfn "${ESMFOLD_PATH}" /Data/public/esmfold_v1
  fi
  cd "${RESULT_ROOT}"
  log "Runtime prepared"
  log "CODE_ROOT=${CODE_ROOT}"
  log "RESULT_ROOT=${RESULT_ROOT}"
  log "MODEL_ROOT=${MODEL_ROOT}"
  log "ZYMCTRL_V2_CKPT=${ZYMCTRL_V2_CKPT}"
  nvidia-smi || true
}

validate_inputs() {
  [[ -f "${ZYMCTRL_V2_CKPT}/clt.pt" ]] || {
    log "Missing ZymCTRL v2 checkpoint: ${ZYMCTRL_V2_CKPT}/clt.pt"
    exit 2
  }
  [[ -f "${EC_FEATURES}" ]] || {
    log "Missing EC features: ${EC_FEATURES}"
    exit 2
  }
  python - <<PY
import pickle
p = "${EC_FEATURES}"
d = pickle.load(open(p, "rb"))
k = next(iter(d))
print(f"EC features: {p}  classes={len(d)}  d_clt={d[k]['d_clt']}  n_layers={d[k]['n_layers']}")
assert int(d[k]["d_clt"]) == 8192, "ec_features.pkl is not from the 8192-dim v2 CLT"
PY
}

run_case_study_if_needed() {
  skip_or_run "${LYSOZYME_LEADS}" try_logged r2e_case_study_v2 "
    cd '${RESULT_ROOT}'
    CUDA_VISIBLE_DEVICES=0 python '${CODE_ROOT}/scripts/12_drug_design_case_study.py' \
      --target ec_lysozyme \
      --model zymctrl \
      --clt '${ZYMCTRL_V2_CKPT}' \
      --ec-features '${EC_FEATURES}' \
      --n '${N_CASE}' \
      --top-k 10 \
      --include-unsteered \
      --out '${LYSOZYME_LEADS}'
  "
}

run_steering_benchmark() {
  local out="${RESULT_ROOT}/results/steering_benchmark/zymctrl_v2_purity_${STAMP}.json"
  try_logged r2d_steering_benchmark_v2 "
    cd '${RESULT_ROOT}'
    CUDA_VISIBLE_DEVICES=0 python '${CODE_ROOT}/scripts/11_steering_benchmark.py' \
      --model zymctrl \
      --clt '${ZYMCTRL_V2_CKPT}' \
      --ec-features '${EC_FEATURES}' \
      --n '${N_BENCH}' \
      --max-new-tokens '${MAX_NEW_TOKENS}' \
      --out '${out}'
  "
  if [[ -s "${out}" ]]; then
    ln -sfn "${out}" "${RESULT_ROOT}/results/steering_benchmark/zymctrl_v2_purity_latest.json"
  fi
}

run_structural_qc() {
  local steered_out="${RESULT_ROOT}/results/drug_design/ec_lysozyme_esmfold_metrics_v2_${STAMP}.json"
  local unsteered_out="${RESULT_ROOT}/results/drug_design/ec_lysozyme_unsteered_esmfold_metrics_v2_${STAMP}.json"
  try_logged r2f_structural_qc_steered_v2 "
    cd '${RESULT_ROOT}'
    CUDA_VISIBLE_DEVICES='${STRUCTURAL_CUDA_DEVICE}' python '${CODE_ROOT}/scripts/13_structural_qc.py' \
      --input '${LYSOZYME_LEADS}' \
      --field leads \
      --backend local \
      --dtype '${STRUCTURAL_DTYPE}' \
      --max-fold 20 \
      --max-structures 20 \
      --out '${steered_out}'
  "
  if [[ -s "${steered_out}" ]]; then
    ln -sfn "${steered_out}" "${RESULT_ROOT}/results/drug_design/ec_lysozyme_esmfold_metrics_v2_latest.json"
  fi
  try_logged r2f_structural_qc_unsteered_v2 "
    cd '${RESULT_ROOT}'
    CUDA_VISIBLE_DEVICES='${STRUCTURAL_CUDA_DEVICE}' python '${CODE_ROOT}/scripts/13_structural_qc.py' \
      --input '${LYSOZYME_LEADS}' \
      --field unsteered_baseline \
      --backend local \
      --dtype '${STRUCTURAL_DTYPE}' \
      --max-fold 20 \
      --max-structures 20 \
      --out '${unsteered_out}'
  "
  if [[ -s "${unsteered_out}" ]]; then
    ln -sfn "${unsteered_out}" "${RESULT_ROOT}/results/drug_design/ec_lysozyme_unsteered_esmfold_metrics_v2_latest.json"
  fi
}

pick_lysozyme_feature() {
  python - <<PY
import pickle, numpy as np
p = "${EC_FEATURES}"
d = pickle.load(open(p, "rb"))
names = list(d)
target = "lysozyme"
layer = 12
mat = np.stack([d[n]["mean"][layer] for n in names])
idx = names.index(target)
z = (mat[idx] - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-6)
feat = int(np.argsort(-z)[0])
print(f"{layer} {feat} {float(z[feat])}")
PY
}

run_causal_ablation() {
  [[ -f "${LYSOZYME_FASTA}" ]] || {
    log "Missing lysozyme reference FASTA: ${LYSOZYME_FASTA}; skipping ablation"
    return 0
  }
  local spec layer feature zscore out
  spec="$(pick_lysozyme_feature)"
  read -r layer feature zscore <<<"${spec}"
  out="${RESULT_ROOT}/results/causal_ablation/zymctrl_v2_lysozyme_L${layer}_F${feature}_${STAMP}.json"
  run_logged r2g_causal_ablation_v2 "
    cd '${RESULT_ROOT}'
    CUDA_VISIBLE_DEVICES=0 python '${CODE_ROOT}/scripts/14_causal_feature_ablation.py' \
      --model zymctrl \
      --clt '${ZYMCTRL_V2_CKPT}' \
      --ec-features '${EC_FEATURES}' \
      --ec-class lysozyme \
      --layer '${layer}' \
      --feature '${feature}' \
      --sequences '${LYSOZYME_FASTA}' \
      --max-sequences 10 \
      --out '${out}'
  "
  ln -sfn "${out}" "${RESULT_ROOT}/results/causal_ablation/zymctrl_v2_lysozyme_latest.json"
  log "Ablated lysozyme feature L${layer} F${feature} z=${zscore}"
}

run_cross_model_conservation() {
  [[ -f "${PROGEN2_CKPT}/clt.pt" ]] || {
    log "Missing ProGen2 checkpoint: ${PROGEN2_CKPT}/clt.pt; skipping conservation"
    return 0
  }
  local out="${RESULT_ROOT}/results/circuit_analysis/cross_model_conservation_v2_${STAMP}.json"
  run_logged r2h_cross_model_conservation_v2 "
    cd '${RESULT_ROOT}'
    CUDA_VISIBLE_DEVICES=0 python '${CODE_ROOT}/scripts/15_cross_model_conservation.py' \
      --model-spec zymctrl='${ZYMCTRL_V2_CKPT}' \
      --model-spec progen2-medium='${PROGEN2_CKPT}' \
      --json '${LYSOZYME_LEADS}' \
      --json-field leads \
      --layers 3 12 30 \
      --max-sequences 10 \
      --max-length 256 \
      --out '${out}'
  "
  ln -sfn "${out}" "${RESULT_ROOT}/results/circuit_analysis/cross_model_conservation_v2_latest.json"
}

write_summary() {
  local out="${RESULT_ROOT}/results/r2_v2_remaining_summary_${STAMP}.json"
  python - <<PY
import json, os, glob
out = "${out}"
summary = {
    "stamp": "${STAMP}",
    "zymctrl_v2_checkpoint": "${ZYMCTRL_V2_CKPT}",
    "ec_features": "${EC_FEATURES}",
    "lysozyme_leads": "${LYSOZYME_LEADS}",
    "latest_outputs": {
        "steering_benchmark": "${RESULT_ROOT}/results/steering_benchmark/zymctrl_v2_purity_latest.json",
        "structural_qc_steered": "${RESULT_ROOT}/results/drug_design/ec_lysozyme_esmfold_metrics_v2_latest.json",
        "structural_qc_unsteered": "${RESULT_ROOT}/results/drug_design/ec_lysozyme_unsteered_esmfold_metrics_v2_latest.json",
        "causal_ablation": "${RESULT_ROOT}/results/causal_ablation/zymctrl_v2_lysozyme_latest.json",
        "cross_model_conservation": "${RESULT_ROOT}/results/circuit_analysis/cross_model_conservation_v2_latest.json",
    },
    "existing": {},
}
for key, path in summary["latest_outputs"].items():
    summary["existing"][key] = os.path.exists(path)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote {out}")
PY
  ln -sfn "${out}" "${RESULT_ROOT}/results/r2_v2_remaining_summary_latest.json"
}

main() {
  log "R2 v2 remaining TODO runner started"
  prepare_runtime
  validate_inputs
  run_case_study_if_needed
  run_steering_benchmark
  run_structural_qc
  run_causal_ablation
  run_cross_model_conservation
  write_summary
  log "R2 v2 remaining TODO runner finished"
}

main "$@"
