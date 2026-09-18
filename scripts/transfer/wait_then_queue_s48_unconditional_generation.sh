#!/usr/bin/env bash
# Wait until MegaScale s29 and the s46x homologous-context expansion no longer
# occupy cards, then freeze and queue unconditional generation. Named cell
# failures on those queues stay marked there and do not withhold this one.
# There is no --depends column: this wrapper is the only gate.
#
# Usage, on the workstation, with H200_POD already set in this shell:
#
#   bash scripts/transfer/wait_then_queue_s48_unconditional_generation.sh
#   bash scripts/transfer/wait_then_queue_s48_unconditional_generation.sh --now
#
# --now skips the wait and launches immediately. The s48 manifest packs cards
# 0-3 and must not launch beside s29 30B or s46x. Do not persist the pod name.
# Do not read hangzhou-compute/config.sh.
set -euo pipefail

LAUNCH_NOW=0
for arg in "$@"; do
  case "${arg}" in
    --now) LAUNCH_NOW=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPFS_PROJECT_ROOT="${GPFS_PROJECT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer}"
HANGZHOU_COMPUTE_ROOT="${HANGZHOU_COMPUTE_ROOT:-${HOME}/hangzhou-compute}"
H200_CLI="${H200_CLI:-${HANGZHOU_COMPUTE_ROOT}/h200}"

S29_STATUS="${GPFS_PROJECT_ROOT}/logs/external_baseline/s29_galactica_instructprotein.status.tsv"
# The s46x snapshot queue does not pass --campaign, so the status basename is
# the manifest stem written by h200_campaign_queue.sh.
S46X_STATUS="${GPFS_PROJECT_ROOT}/logs/external_baseline/campaign_s46_homologue_expansion.status.tsv"
POLL_SECONDS="${POLL_SECONDS:-60}"
MANIFEST_NAME="campaign_s48_unconditional_generation.tsv"
SNAPSHOT_KEY="s48"

log() { printf '[s48-wait] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

if [ -z "${H200_POD:-}" ]; then
  echo "H200_POD must be set in this shell and is not persisted" >&2
  exit 2
fi

# Occupancy gate, not a cleanliness certificate. Named cell failures stay
# marked on that queue and do not withhold the next experiment.
require_queue_finished() {
  local status="$1"
  if [ ! -f "${status}" ]; then
    return 1
  fi
  local failures pending running
  failures="$(awk -F'\t' '$1=="# FAILURES"{print $2; exit}' "${status}")"
  pending="$(sed -n 's/^# tally.*pending=\([0-9]*\).*/\1/p' "${status}" | tail -n1)"
  running="$(sed -n 's/^# tally.*running=\([0-9]*\).*/\1/p' "${status}" | tail -n1)"
  if [ "${pending}" = "0" ] && [ "${running}" = "0" ]; then
    if [ -n "${failures}" ] && [ "${failures}" != "0" ]; then
      log "$(basename "${status}") finished with marked failures: ${failures}"
    fi
    return 0
  fi
  log "$(basename "${status}") still occupying: FAILURES=${failures:-?} pending=${pending:-?} running=${running:-?}"
  return 1
}

STATUS_S29="$(mktemp)"
STATUS_S46X="$(mktemp)"
trap 'rm -f "${STATUS_S29}" "${STATUS_S46X}"' EXIT
if [ "${LAUNCH_NOW}" -eq 1 ]; then
  log "LAUNCH_NOW: skip wait; manifest uses cards 0-3 and must not share the allocation"
else
log "polling ${S29_STATUS} and ${S46X_STATUS} every ${POLL_SECONDS}s via the selected pod"
while true; do
  set +e
  "${H200_CLI}" exec -- cat "${S29_STATUS}" > "${STATUS_S29}" 2>/dev/null
  fetch_s29=$?
  "${H200_CLI}" exec -- cat "${S46X_STATUS}" > "${STATUS_S46X}" 2>/dev/null
  fetch_s46x=$?
  set -e
  if [ "${fetch_s29}" -ne 0 ]; then
    log "s29 status not readable yet"
    sleep "${POLL_SECONDS}"
    continue
  fi
  if [ "${fetch_s46x}" -ne 0 ]; then
    log "s46x status not readable yet"
    sleep "${POLL_SECONDS}"
    continue
  fi
  set +e
  require_queue_finished "${STATUS_S29}"
  ready_s29=$?
  require_queue_finished "${STATUS_S46X}"
  ready_s46x=$?
  set -e
  if [ "${ready_s29}" -eq 0 ] && [ "${ready_s46x}" -eq 0 ]; then
    log "s29 and s46x no longer occupy cards"
    break
  fi
  sleep "${POLL_SECONDS}"
done
fi

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
FREEZE_META="${GPFS_PROJECT_ROOT}/logs/external_baseline/s48_unconditional_generation.freeze.txt"
"${H200_CLI}" exec -- bash -lc "
  mkdir -p '$(dirname "${FREEZE_META}")'
  printf 'RUN_ID=%s\nSNAPSHOT_DIR=%s\n' '${RUN_ID}' '${SNAPSHOT_DIR}' > '${FREEZE_META}'
"
log "froze RUN_ID=${RUN_ID}"
log "snapshot ${SNAPSHOT_DIR}"

MANIFEST="${SNAPSHOT_DIR}/scripts/transfer/${MANIFEST_NAME}"
if ! "${H200_CLI}" exec -- test -f "${MANIFEST}"; then
  echo "snapshot copy of ${MANIFEST_NAME} is missing on the selected pod: ${MANIFEST}" >&2
  exit 2
fi
log "launching campaign queue from the snapshot copy of the manifest"
"${H200_CLI}" exec -- bash -lc "
  setsid nohup bash '${SNAPSHOT_DIR}/scripts/transfer/h200_campaign_queue.sh' \
    --manifest '${MANIFEST}' \
    --snapshot ${SNAPSHOT_KEY}='${SNAPSHOT_DIR}' \
    > /dev/null 2>&1 < /dev/null &
  disown
  echo QUEUE_LAUNCHED
"
log "QUEUE_LAUNCHED requested; evidence is the status file appearing under logs/external_baseline/"
