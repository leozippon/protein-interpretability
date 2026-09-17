#!/usr/bin/env bash
# Wait for MegaScale s29_galactica_instructprotein, then freeze and queue the
# homologous-context expansion wave. There is no --depends column: this wrapper
# is the only gate.
#
# Usage, on the workstation, with H200_POD already set in this shell:
#
#   bash scripts/transfer/wait_then_queue_s46_homologue_expansion.sh
#
# Do not persist the pod name. Do not read hangzhou-compute/config.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPFS_PROJECT_ROOT="${GPFS_PROJECT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer}"
HANGZHOU_COMPUTE_ROOT="${HANGZHOU_COMPUTE_ROOT:-${HOME}/hangzhou-compute}"
H200_CLI="${H200_CLI:-${HANGZHOU_COMPUTE_ROOT}/h200}"

F16_RUN_ID="20260826143603_cffa47b0ef19"
F16_COHORT_DIGEST="33707cee59c523dec945f9cc6f815bf5e5a1610eeff6e40ca2f16ce3134a7532"
S29_STATUS="${GPFS_PROJECT_ROOT}/logs/external_baseline/campaign_s29_galactica_instructprotein.status.tsv"
S29_THIRTY="s29_galactica-30b"
POLL_SECONDS="${POLL_SECONDS:-60}"
MANIFEST_NAME="campaign_s46_homologue_expansion.tsv"
SNAPSHOT_KEY="s46x"

log() { printf '[s46x-wait] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

if [ -z "${H200_POD:-}" ]; then
  echo "H200_POD must be set in this shell and is not persisted" >&2
  exit 2
fi

require_s29_ready() {
  local status="$1"
  if [ ! -f "${status}" ]; then
    return 1
  fi
  local failures no_record pending running thirty bad
  failures="$(awk -F'\t' '$1=="# FAILURES"{print $2; exit}' "${status}")"
  no_record="$(awk -F'\t' '$1=="# NO-RECORD"{print $2; exit}' "${status}")"
  failures="${failures%%:*}"
  no_record="${no_record%%:*}"
  pending="$(sed -n 's/^# tally.*pending=\([0-9]*\).*/\1/p' "${status}" | tail -n1)"
  running="$(sed -n 's/^# tally.*running=\([0-9]*\).*/\1/p' "${status}" | tail -n1)"
  thirty="$(awk -F'\t' -v name="${S29_THIRTY}" '$2==name{print $4; exit}' "${status}")"
  bad="$(awk -F'\t' 'NR>1 && $1 !~ /^#/ && ($4=="exited-nonzero" || $4=="refused-busy-gpu"){print $2"="$4}' "${status}")"
  if [ -n "${bad}" ]; then
    log "s29 has a refused or nonzero cell: ${bad}"
    return 2
  fi
  if [ "${failures}" = "0" ] && [ "${no_record}" = "0" ] && [ "${pending}" = "0" ] && [ "${running}" = "0" ] && { [ "${thirty}" = "exited-ok" ] || [ "${thirty}" = "skipped-complete" ]; }; then
    return 0
  fi
  log "s29 not ready: FAILURES=${failures:-?} NO-RECORD=${no_record:-?} pending=${pending:-?} running=${running:-?} ${S29_THIRTY}=${thirty:-absent}"
  return 1
}

STATUS_COPY="$(mktemp)"
trap 'rm -f "${STATUS_COPY}"' EXIT
log "polling ${S29_STATUS} every ${POLL_SECONDS}s via the selected pod"
while true; do
  set +e
  "${H200_CLI}" exec -- cat "${S29_STATUS}" > "${STATUS_COPY}" 2>/dev/null
  fetch=$?
  set -e
  if [ "${fetch}" -ne 0 ]; then
    log "s29 status not readable yet"
    sleep "${POLL_SECONDS}"
    continue
  fi
  set +e
  require_s29_ready "${STATUS_COPY}"
  ready=$?
  set -e
  if [ "${ready}" -eq 0 ]; then
    log "s29 finished cleanly; ${S29_THIRTY} is complete"
    break
  fi
  if [ "${ready}" -eq 2 ]; then
    echo "refusing to launch s46x after an s29 failure" >&2
    exit 2
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
FREEZE_META="${GPFS_PROJECT_ROOT}/logs/external_baseline/s46x_homologue_expansion.freeze.txt"
mkdir -p "$(dirname "${FREEZE_META}")"
{
  printf 'RUN_ID=%s\n' "${RUN_ID}"
  printf 'SNAPSHOT_DIR=%s\n' "${SNAPSHOT_DIR}"
} > "${FREEZE_META}"
log "froze RUN_ID=${RUN_ID}"
log "snapshot ${SNAPSHOT_DIR}"

RESULTS_DIR="${GPFS_PROJECT_ROOT}/results/external_baseline/${RUN_ID}"
mkdir -p "${RESULTS_DIR}"
F16_COHORT=""
for candidate in \
  "${GPFS_PROJECT_ROOT}/results/external_baseline/${F16_RUN_ID}/cohort.json" \
  "${GPFS_PROJECT_ROOT}/results/external_baseline/${F16_RUN_ID}/context_homologue/cohort.json"
do
  if [ -f "${candidate}" ]; then
    F16_COHORT="${candidate}"
    break
  fi
done
if [ -z "${F16_COHORT}" ]; then
  echo "F16 cohort not found under ${GPFS_PROJECT_ROOT}/results/external_baseline/${F16_RUN_ID}" >&2
  exit 2
fi
cp -f -- "${F16_COHORT}" "${RESULTS_DIR}/cohort.json"
observed="$(sha256sum "${RESULTS_DIR}/cohort.json" | awk '{print $1}')"
if [ "${observed}" != "${F16_COHORT_DIGEST}" ]; then
  echo "staged cohort hashes to ${observed}, expected ${F16_COHORT_DIGEST}" >&2
  exit 2
fi
log "staged F16 cohort into ${RESULTS_DIR}/cohort.json (sha256 verified)"

MANIFEST="${SNAPSHOT_DIR}/scripts/transfer/${MANIFEST_NAME}"
if [ ! -f "${MANIFEST}" ]; then
  echo "snapshot copy of ${MANIFEST_NAME} is missing: ${MANIFEST}" >&2
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
