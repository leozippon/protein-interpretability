#!/usr/bin/env bash
# After s48 unconditional generation no longer occupies cards, freeze and run
# the EXP-R2-247 inventory on CPU. Coefficients wait until pending generation
# analysis cells exist; this wrapper does not invent them and does not take a GPU.
#
# Usage, with H200_POD already set in this shell:
#
#   bash scripts/transfer/wait_then_run_s50_cross_measure_association.sh
#
# Do not persist the pod name. Do not read hangzhou-compute/config.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPFS_PROJECT_ROOT="${GPFS_PROJECT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer}"
HANGZHOU_COMPUTE_ROOT="${HANGZHOU_COMPUTE_ROOT:-${HOME}/hangzhou-compute}"
H200_CLI="${H200_CLI:-${HANGZHOU_COMPUTE_ROOT}/h200}"
S48_STATUS="${GPFS_PROJECT_ROOT}/logs/external_baseline/campaign_s48_unconditional_generation.status.tsv"
POLL_SECONDS="${POLL_SECONDS:-60}"
TRANSFER_PYTHON="${TRANSFER_PYTHON:-/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer/runtimes/ct-20260905/bin/python}"

log() { printf '[s50-wait] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

if [ -z "${H200_POD:-}" ]; then
  echo "H200_POD must be set in this shell and is not persisted" >&2
  exit 2
fi

require_queue_finished() {
  local status="$1"
  if [ ! -f "${status}" ]; then
    return 1
  fi
  local pending running
  pending="$(sed -n 's/^# tally.*pending=\([0-9]*\).*/\1/p' "${status}" | tail -n1)"
  running="$(sed -n 's/^# tally.*running=\([0-9]*\).*/\1/p' "${status}" | tail -n1)"
  if [ "${pending}" = "0" ] && [ "${running}" = "0" ]; then
    return 0
  fi
  log "$(basename "${status}") still occupying: pending=${pending:-?} running=${running:-?}"
  return 1
}

STATUS_COPY="$(mktemp)"
trap 'rm -f "${STATUS_COPY}"' EXIT
log "polling ${S48_STATUS} every ${POLL_SECONDS}s via the selected pod"
while true; do
  set +e
  "${H200_CLI}" exec -- cat "${S48_STATUS}" > "${STATUS_COPY}" 2>/dev/null
  fetch=$?
  set -e
  if [ "${fetch}" -ne 0 ]; then
    log "s48 status not readable yet"
    sleep "${POLL_SECONDS}"
    continue
  fi
  set +e
  require_queue_finished "${STATUS_COPY}"
  ready=$?
  set -e
  if [ "${ready}" -eq 0 ]; then
    log "s48 no longer occupies cards"
    break
  fi
  sleep "${POLL_SECONDS}"
done

log "freezing a new snapshot with --pin HEAD --freeze-only"
FREEZE_LOG="$(mktemp)"
set +e
(
  cd "${REPO_ROOT}"
  bash scripts/transfer/run_transfer_h200.sh --pin HEAD --freeze-only
) > "${FREEZE_LOG}" 2>&1
freeze_status=$?
set -e
if [ "${freeze_status}" -ne 0 ]; then
  echo "freeze-only failed; log at ${FREEZE_LOG}" >&2
  tail -n 40 "${FREEZE_LOG}" >&2
  exit "${freeze_status}"
fi
RUN_ID="$(sed -n 's/^RUN_ID=//p' "${FREEZE_LOG}" | tail -n1)"
SNAPSHOT_DIR="$(sed -n 's/^SNAPSHOT_DIR=//p' "${FREEZE_LOG}" | tail -n1)"
if [ -z "${RUN_ID}" ] || [ -z "${SNAPSHOT_DIR}" ]; then
  echo "freeze-only did not print RUN_ID= and SNAPSHOT_DIR=; refusing" >&2
  tail -n 40 "${FREEZE_LOG}" >&2
  exit 2
fi
OUT="${GPFS_PROJECT_ROOT}/results/external_baseline/${RUN_ID}/s50_cross_measure"
log "running CPU inventory from the snapshot"
"${H200_CLI}" exec -- bash -lc "
  '${TRANSFER_PYTHON}' '${SNAPSHOT_DIR}/scripts/transfer/50_cross_measure_association.py' \
    --stage inventory --device cpu --out '${OUT}'
"
log "inventory written under ${OUT}; associate waits until U2–U5 analysis cells exist"
