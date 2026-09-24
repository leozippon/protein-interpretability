#!/usr/bin/env bash
# Wait for the declared cards of one allocation to fall idle, then dispatch one
# single-mutant stability extraction manifest through the in-pod campaign queue.
#
# It exists because the whole allocation was busy with other campaigns when this
# gate's extraction became ready: every GPU cell would have been recorded
# refused-busy-gpu, which is the queue behaving correctly. This wrapper waits for
# the cards instead of lowering the gate. It never changes the queue's idle
# threshold, never co-tenants a card, and never forces past a held lock.
#
# The wait loop runs here, on the workstation, exactly as the other
# wait-then-queue wrappers in this directory do; only the final queue launch
# crosses into the pod. Run it detached (`nohup ... &`) if the wait may be long.
#
# Usage, with H200_POD already set in this shell:
#
#   bash scripts/transfer/wait_then_queue_gate_stability.sh \
#       --snapshot-dir <gpfs snapshot dir> \
#       --manifest campaign_gate_stability_l2a.tsv --cards 0,1
#   ... --now     dispatch immediately without waiting for the cards
#   ... --manifest-dir <gpfs dir>
#                 read the manifest from this directory instead of the snapshot's
#                 own scripts/transfer. A wave declared after a snapshot was
#                 frozen must not be written into that snapshot: the controller
#                 refuses to reuse a run-id whose GPFS directory holds different
#                 content than the code it was derived from. The code that runs
#                 is still the snapshot's.
#
# `QUEUE_LAUNCHED` is not evidence that anything ran: the evidence is the status
# file appearing at logs/external_baseline/<manifest basename>.status.tsv on
# GPFS. The pod name stays in the invoking shell and is written nowhere.
set -euo pipefail

SNAPSHOT_DIR=""
MANIFEST_NAME=""
MANIFEST_DIR=""
CARDS=""
LAUNCH_NOW=0
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"

#: The queue's own idle threshold, in MiB, quoted from h200_campaign_queue.sh.
#: It is a constant here so that this wrapper cannot be used to relax it.
IDLE_MIB=1000

while [ "$#" -gt 0 ]; do
  case "$1" in
    --snapshot-dir) SNAPSHOT_DIR="$2"; shift 2 ;;
    --manifest) MANIFEST_NAME="$2"; shift 2 ;;
    --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
    --cards) CARDS="$2"; shift 2 ;;
    --now) LAUNCH_NOW=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${H200_POD:-}" ]; then
  echo "H200_POD must be set in this shell and is not persisted" >&2
  exit 2
fi
if [ -z "${SNAPSHOT_DIR}" ] || [ -z "${MANIFEST_NAME}" ] || [ -z "${CARDS}" ]; then
  echo "--snapshot-dir, --manifest and --cards are all required" >&2
  exit 2
fi
case "${MANIFEST_NAME}" in
  */*|"") echo "--manifest takes a basename inside the snapshot" >&2; exit 2 ;;
esac

HANGZHOU_COMPUTE_ROOT="${HANGZHOU_COMPUTE_ROOT:-${HOME}/hangzhou-compute}"
H200_CLI="${H200_CLI:-${HANGZHOU_COMPUTE_ROOT}/h200}"
MANIFEST="${MANIFEST_DIR:-${SNAPSHOT_DIR}/scripts/transfer}/${MANIFEST_NAME}"
CAMPAIGN="${MANIFEST_NAME%.tsv}"

log() { printf '[gate-stability-wait] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

busy_cards() {
  local snapshot card used out=""
  snapshot="$("${H200_CLI}" exec -- nvidia-smi --query-gpu=index,memory.used \
              --format=csv,noheader,nounits 2>/dev/null)" || return 3
  for card in ${CARDS//,/ }; do
    used="$(printf '%s\n' "${snapshot}" | awk -F', ' -v c="${card}" '$1==c {print $2}')"
    if [ -z "${used}" ]; then
      out="${out} ${card}(absent)"
    elif [ "${used}" -gt "${IDLE_MIB}" ]; then
      out="${out} ${card}(${used}MiB)"
    fi
  done
  printf '%s' "${out}"
}

log "manifest ${MANIFEST_NAME}, cards ${CARDS}, poll ${POLL_SECONDS}s, ceiling ${MAX_WAIT_SECONDS}s"
waited=0
while [ "${LAUNCH_NOW}" -eq 0 ]; do
  set +e
  busy="$(busy_cards)"
  probe=$?
  set -e
  if [ "${probe}" -eq 3 ]; then
    log "card state not readable; retrying"
  elif [ -z "${busy}" ]; then
    log "all declared cards idle"
    break
  else
    log "waiting, busy:${busy}"
  fi
  if [ "${waited}" -ge "${MAX_WAIT_SECONDS}" ]; then
    log "ceiling reached with cards busy:${busy:-unknown}; not launching"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
  waited=$((waited + POLL_SECONDS))
done

log "launching the campaign queue in the selected pod"
"${H200_CLI}" exec -- bash -lc "
  test -f '${MANIFEST}' || { echo 'MANIFEST_ABSENT'; exit 2; }
  setsid nohup bash '${SNAPSHOT_DIR}/scripts/transfer/h200_campaign_queue.sh' \
    --manifest '${MANIFEST}' \
    --snapshot cc='${SNAPSHOT_DIR}' \
    > /dev/null 2>&1 < /dev/null &
  disown
  echo QUEUE_LAUNCHED
"
log "QUEUE_LAUNCHED requested; evidence is logs/external_baseline/${CAMPAIGN}.status.tsv"
