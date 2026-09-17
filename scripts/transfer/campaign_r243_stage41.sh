#!/usr/bin/env bash
set -euo pipefail

# EXP-R2-243's stage 41, run directly in the pod instead of as a queue row.
#
# WHY IT IS NOT A QUEUE ROW. `41_context_information_bootstrap.py` takes no
# `--device`: it loads no model, touches no GPU and reads only statistics
# already on disk. `h200_campaign_queue.sh` injects `--device` on every row it
# launches, `cpu` rows included, so a cpu row naming this stage exits 2 with
# `unrecognized arguments: --device cpu` before the stage reads anything --
# which is what both stage-41 rows of the manifest's first revision did. The
# alternative, teaching the runner to omit `--device` for cpu rows, would break
# every cpu row whose stage *requires* it (`native_dms_extension.py analyse`,
# `text_aa_dms.py`), so the stage is run from here with `CUDA_VISIBLE_DEVICES`
# empty and nothing else changed.
#
# WHAT IT RUNS, AND WHY THE ARGUMENT LIST IS WIDER THAN THE FAILED ROWS'.
# Both keys are analysed at the manifest's declared settings: seed 20260820,
# 2000 paired bootstrap, `--alpha-sweep 1.0`, `--arms rita-xl`,
# `--expected-arms rita-xl`, into the label directory the first revision
# declared for each key. Each key's eight sidecars are read from that key's own
# results tree, which is the run id its snapshot directory is named by.
#
# The companion `cohort_*.json` and `reference_*.json` of every block are
# supplied as well, which the failed rows did not declare. That is not a second
# convention, it is the configuration the recorded envelope was measured in:
# group assignments are not persisted in the sidecar, so stage 41 recomputes the
# near-duplicate groups from the cohort records and falls back to singleton
# groups where they are absent -- a fallback it writes into every affected record
# as a declared limitation, and one that narrows the interval exactly where group
# dependence is strongest. EXP-R2-240's stage-41 run, which produced the recorded
# `[+1.1942, +1.5878]` for this arm, supplied both: its report's
# `metadata.configuration` records the cohort_json and reference_json lists.
# Omitting them here would move the interval for a reason the rendering has
# nothing to do with, and the before/after comparison this campaign exists to
# make would not be one.
#
# READINESS AND FAILURE RULES. Each key's eight blocks must each carry exactly one
# `power_*.records.npz`, one `cohort_*.json` and one `reference_*.json` before that
# key is analysed: a block carrying several of a kind is refused outright, because
# the names carry each block's own digest and a cell that drew something else must
# not be paired against a block it does not share. A block that is merely absent is
# waited for, with a bound; a key that is not ready by then is refused rather than
# analysed over a partial set. A stage-41 invocation that exits nonzero stops the
# harness. The DONE file records both exit codes.
#
# Usage, on the controller, after freezing the campaign's snapshot:
#
#   H200_POD=<pod> ~/hangzhou-compute/ssh_tunnel/h200_pod_exec.sh -- bash -lc "
#     setsid nohup bash '<native-snapshot>/scripts/transfer/campaign_r243_stage41.sh' \
#       --snapshot-native '<native-snapshot>' \
#       --snapshot-base '<base-snapshot>' \
#       > /dev/null 2>&1 < /dev/null &
#     disown
#     echo STAGE41_LAUNCHED
#   "
#
# The evidence is the DONE file under the native key's `_stage41` label directory,
# not the `echo`, for the reason the queue runner's header records.

GPFS_PROJECT_ROOT="${GPFS_PROJECT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer}"
SEED=20260820
N_BOOTSTRAP=2000
ALPHA_SWEEP=1.0
ARM=rita-xl
BLOCKS=(0 1 2 3 4 5 6 7)
WAIT_SECONDS=900
PYTHON="${GPFS_PROJECT_ROOT}/runtimes/ct-20260905/bin/python"
SNAPSHOT_NATIVE=""
SNAPSHOT_BASE=""
NATIVE_PREFIX="r243_rita-xl"
BASE_PREFIX="r243_rita-xl_base"
NATIVE_LABEL="r243_rita-xl_stage41"
BASE_LABEL="r243_rita-xl_base_stage41"

usage() {
  sed -n '/^# Usage, on the controller/,/^# The evidence is the DONE file/p' "${BASH_SOURCE[0]}" \
    | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --snapshot-native) SNAPSHOT_NATIVE="$2"; shift 2 ;;
    --snapshot-base) SNAPSHOT_BASE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[r243-stage41] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

for key in native base; do
  if [ "${key}" = "native" ]; then dir="${SNAPSHOT_NATIVE}"; else dir="${SNAPSHOT_BASE}"; fi
  [ -n "${dir}" ] || { echo "--snapshot-${key} is required" >&2; exit 2; }
  [ -f "${dir}/scripts/transfer/41_context_information_bootstrap.py" ] || {
    echo "snapshot ${key} does not carry scripts/transfer/41_context_information_bootstrap.py: ${dir}" >&2
    exit 2; }
done
[ -x "${PYTHON}" ] || { echo "no executable interpreter at ${PYTHON}" >&2; exit 2; }

NATIVE_RESULTS="${GPFS_PROJECT_ROOT}/results/external_baseline/$(basename "${SNAPSHOT_NATIVE}")"
BASE_RESULTS="${GPFS_PROJECT_ROOT}/results/external_baseline/$(basename "${SNAPSHOT_BASE}")"

# The one file of one kind under one block directory, or a nonzero status: 1 when
# none matched (not ready yet) and 2 when several did (refused).
block_files() {
  local results="$1" prefix="$2" block="$3" kind="$4"
  local dir="${results}/${prefix}_b${block}" path found=()
  for path in "${dir}/${kind}_"*; do
    [ -e "${path}" ] || break
    found+=("${path}")
  done
  case "${#found[@]}" in
    0) return 1 ;;
    1) printf '%s\n' "${found[0]}" ;;
    *)
      {
        echo "block ${prefix}_b${block} under ${results} carries ${#found[@]} ${kind}_* files:"
        printf '  %s\n' "${found[@]}"
      } >&2
      return 2
      ;;
  esac
}

# A block carrying several files of one kind is a refusal rather than a wait, so
# it is checked once. A missing file is the wait condition.
require_one_of_each() {
  local results="$1" prefix="$2" block kind status
  for block in "${BLOCKS[@]}"; do
    for kind in power cohort reference; do
      block_files "${results}" "${prefix}" "${block}" "${kind}" >/dev/null || {
        status=$?
        [ "${status}" -eq 2 ] && exit 2
      }
    done
  done
  return 0
}

ready_blocks() {
  local results="$1" prefix="$2" block kind count=0 ok
  for block in "${BLOCKS[@]}"; do
    ok=1
    for kind in power cohort reference; do
      block_files "${results}" "${prefix}" "${block}" "${kind}" >/dev/null 2>&1 || ok=0
    done
    [ "${ok}" -eq 1 ] && count=$((count + 1))
  done
  printf '%s\n' "${count}"
}

wait_ready() {
  local key="$1" results="$2" prefix="$3" deadline=$((SECONDS + WAIT_SECONDS)) ready
  require_one_of_each "${results}" "${prefix}"
  while :; do
    ready="$(ready_blocks "${results}" "${prefix}")"
    log "${key}: ${ready}/8 blocks carry a records sidecar, a cohort and a reference"
    [ "${ready}" -eq 8 ] && return 0
    [ "${SECONDS}" -lt "${deadline}" ] || {
      echo "${key}: only ${ready}/8 blocks ready after ${WAIT_SECONDS}s under ${results}; refusing" >&2
      return 4; }
    sleep 30
  done
}

analyse() {
  local key="$1" snapshot="$2" results="$3" prefix="$4" label="$5" block
  local -a sidecars=() cohorts=() references=()
  local out
  for block in "${BLOCKS[@]}"; do
    sidecars+=("$(block_files "${results}" "${prefix}" "${block}" power)")
    cohorts+=("$(block_files "${results}" "${prefix}" "${block}" cohort)")
    references+=("$(block_files "${results}" "${prefix}" "${block}" reference)")
  done
  out="${results}/${label}"
  mkdir -p "${out}"
  log "${key}: invoking stage 41 over 8 sidecars into ${out}"
  CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${PYTHON}" "${snapshot}/scripts/transfer/41_context_information_bootstrap.py" \
      --sidecar "${sidecars[@]}" \
      --cohort-json "${cohorts[@]}" \
      --reference-json "${references[@]}" \
      --arms "${ARM}" \
      --expected-arms "${ARM}" \
      --seed "${SEED}" \
      --n-bootstrap "${N_BOOTSTRAP}" \
      --alpha-sweep "${ALPHA_SWEEP}" \
      --out "${out}" \
      --report-name context_information_bootstrap.json
}

DONE="${NATIVE_RESULTS}/${NATIVE_LABEL}/DONE"
mkdir -p "$(dirname "${DONE}")"
native_status=1
base_status=1
set +e
wait_ready native "${NATIVE_RESULTS}" "${NATIVE_PREFIX}" && \
  analyse native "${SNAPSHOT_NATIVE}" "${NATIVE_RESULTS}" "${NATIVE_PREFIX}" "${NATIVE_LABEL}"
native_status=$?
wait_ready base "${BASE_RESULTS}" "${BASE_PREFIX}" && \
  analyse base "${SNAPSHOT_BASE}" "${BASE_RESULTS}" "${BASE_PREFIX}" "${BASE_LABEL}"
base_status=$?
set -e

{
  echo "native_status=${native_status}"
  echo "base_status=${base_status}"
  echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${DONE}"
log "native_status=${native_status} base_status=${base_status}; wrote ${DONE}"
[ "${native_status}" -eq 0 ] && [ "${base_status}" -eq 0 ]
