#!/usr/bin/env bash
set -euo pipefail

# In-pod CPU lane for the global-but-additive sequence-context gate.
#
# It runs fit_global_context_gate.py over a list of roster arms against the
# retained label-free extraction artefacts of the admitted pairwise fit. No model
# is loaded, no forward pass runs and no GPU is touched: every cell is a refit
# over arrays that already exist, so lanes are sized in CPU threads rather than
# cards. The stage itself verifies every frozen input digest and refuses a run
# whose rows or folds differ from the admitted per-arm fit.
#
# Usage, from inside a pod, against a frozen snapshot:
#   bash <snapshot>/scripts/transfer/run_global_context_gate_lane.sh \
#        <output-root> <parallel-arms> <arm> [arm...]
#
# It writes one record per arm to <output-root>/fits/gate_<arm>.json, one log per
# arm to <output-root>/logs/gate_<arm>.log, and one host resource receipt per
# lane invocation to <output-root>/logs/host_state_<timestamp>.txt.

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <output-root> <parallel-arms> <arm> [arm...]" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
PARALLEL="$2"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${HERE}/../.." && pwd)"
# shellcheck source=/dev/null
. "${HERE}/h200_env.sh"

# Frozen declarations. The plan content digest is the one
# docs/D1_PAIRWISE_EPISTASIS_RESULTS.md states for the label-free extraction
# plan; the declaration digest is the one docs/D1_GATE_GLOBAL_CONTEXT.md froze
# before this gate was fitted. The stage refuses to run if either disagrees with
# the artefact it is handed.
PLAN_CONTENT_SHA256="${PLAN_CONTENT_SHA256:-e338420f5df70143ccfc8d16ec5479a67b330ec35d36ea7c5965ea31a59e179c}"
DECLARATION_SHA256="${DECLARATION_SHA256:-b3687416edb17d64108705738398c3d260d52e3ea91f3fd4e723e36f349788e6}"

INDEL_EXCLUSION="${INDEL_EXCLUSION:-${OUTPUT_ROOT}/indel_exclusion.json}"
SUPPORTS="${SUPPORTS:-all,q,all_no_indel}"

INPUTS="${TRANSFER_PROJECT_ROOT}/data/pairwise_epistasis"
EXTRACTION="${TRANSFER_PROJECT_ROOT}/results/pairwise_epistasis_20260924/extraction"
FLAGSHIP="${TRANSFER_PROJECT_ROOT}/results/pairwise_epistasis_20260924/fits"
FITS="${OUTPUT_ROOT}/fits"
LOGS="${OUTPUT_ROOT}/logs"
mkdir -p "${FITS}" "${LOGS}"

REQUIRED=("${INPUTS}/extraction_plan.json" "${INPUTS}/cohort.json"
          "${INPUTS}/baseline_q.json" "${INPUTS}/profile_features.npz"
          "${EXTRACTION}" "${FLAGSHIP}")
# A _no_indel support is the declared insertion/deletion-construct sensitivity
# and cannot be run without the declaration it is defined by.
case "${SUPPORTS}" in
  *_no_indel*) REQUIRED+=("${INDEL_EXCLUSION}") ;;
esac
for path in "${REQUIRED[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "missing required input: ${path}" >&2
    exit 2
  fi
done

RECEIPT="${LOGS}/host_state_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "code_root=${CODE_ROOT}"
  echo "python=${TRANSFER_PYTHON}"
  echo "arms=$*"
  echo "supports=${SUPPORTS}"
  echo "indel_exclusion=${INDEL_EXCLUSION}"
  echo "parallel=${PARALLEL}"
  echo "threads_per_arm=${OMP_NUM_THREADS:-8}"
  echo "nproc=$(nproc)"
  echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
  free -g || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
} > "${RECEIPT}"
echo "host resource receipt ${RECEIPT}"

run_arm() {
  local arm="$1"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  "${TRANSFER_PYTHON}" "${CODE_ROOT}/scripts/transfer/fit_global_context_gate.py" \
    --plan "${INPUTS}/extraction_plan.json" \
    --expect-plan-sha256 "${PLAN_CONTENT_SHA256}" \
    --cohort "${INPUTS}/cohort.json" \
    --baseline-q "${INPUTS}/baseline_q.json" \
    --profiles "${INPUTS}/profile_features.npz" \
    --extraction "${EXTRACTION}" \
    --flagship-fit "${FLAGSHIP}/fit_${arm}.json" \
    --expect-declaration-sha256 "${DECLARATION_SHA256}" \
    --arm "${arm}" \
    --supports "${SUPPORTS}" \
    --indel-exclusion "${INDEL_EXCLUSION}" \
    --out "${FITS}/gate_${arm}.json" \
    > "${LOGS}/gate_${arm}.log" 2>&1
  local status=$?
  echo "arm=${arm} exit=${status}"
  return "${status}"
}
export -f run_arm
export TRANSFER_PYTHON CODE_ROOT INPUTS EXTRACTION FLAGSHIP FITS LOGS \
       PLAN_CONTENT_SHA256 DECLARATION_SHA256 OMP_NUM_THREADS SUPPORTS INDEL_EXCLUSION

FAILURES=0
printf '%s\n' "$@" | xargs -I{} -P "${PARALLEL}" bash -c 'run_arm "$@"' _ {} || FAILURES=1
echo "# FAILURES ${FAILURES}"
exit "${FAILURES}"
