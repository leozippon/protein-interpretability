#!/usr/bin/env bash
set -euo pipefail

# Driver for an EXTERNAL-BASELINE stage on the H200 -- a stage that measures a
# checkpoint which is not a panel arm, and therefore cannot name a registered
# stage and cannot reach run_transfer_h200.sh's scheduling path.
#
# Why this file is committed, when every other driver in this repository lives
# under ignored logs/drivers/. EXP-R2-132 -- the first result of the current main
# line -- was launched by hand. Its code snapshot is on GPFS with a correct
# 43-file manifest, so the freeze was genuinely performed; what does not exist
# anywhere is the dispatch: no driver, no controller log, no INVOCATIONS record,
# no shell history. The parameters survive only because the stage writes
# vars(args) into its own artefact, and vars(args) cannot distinguish a flag
# typed at its default from one that was omitted. A driver under logs/ is
# git-ignored AND mutagen-ignored, so it is neither committed nor synchronised,
# which is precisely how that happened. This one is inside the freeze baseline,
# so the launcher is part of the run's identity -- editing it changes CODE_HASH,
# which is correct.
#
# The freeze is NOT reimplemented here. run_transfer_h200.sh --freeze-only does
# it and prints RUN_ID and SNAPSHOT_DIR (Appendix B rule 12: one declaration,
# imported, never reimplemented).
#
# Privacy: no pod name is written here. Export H200_POD before invoking.
#
# Usage:
#   export H200_POD=<running-pod-name>
#   scripts/transfer/run_external_baseline_h200.sh \
#       --stage 16_fitness_recovery.py \
#       --label progenmech_stratified \
#       --gpu 0 \
#       -- --sampling progenmech_stratified --variants 1000
#
#   Repeat --gpu/--label pairs by invoking once per condition; each condition
#   gets its own results directory, because a resume key has no condition axis
#   and two conditions in one root would overwrite each other.
#
#   To run several conditions of ONE comparison concurrently, freeze once and
#   hand every invocation the same snapshot:
#
#     eval "$(scripts/transfer/run_transfer_h200.sh --freeze-only)"   # RUN_ID, SNAPSHOT_DIR
#     scripts/transfer/run_external_baseline_h200.sh --run-id "$RUN_ID" \
#         --snapshot-dir "$SNAPSHOT_DIR" --stage ... --label ... --gpu 0 -- ... &
#
#   Two reasons, one operational and one scientific. Four controllers freezing at
#   once collide on the shared relay's single temp script path -- a hazard
#   EXP-R2-122 recorded and a 20-second stagger did not fix, because a push
#   occupies the relay for minutes. And the arms of one comparison must run the
#   same code: a CLT and a PLT frozen from two snapshots are not the controlled
#   comparison they are reported as. Reusing a snapshot is refused unless it is
#   present on the pod, so this cannot silently run against nothing.

H200_ACCESS_ROOT="${H200_ACCESS_ROOT:-${HOME}/hangzhou-remote}"
H200_SYNC="${H200_SYNC:-${H200_ACCESS_ROOT}/ssh_tunnel/h200_sync.sh}"
H200_POD_BASH="${H200_POD_BASH:-${H200_ACCESS_ROOT}/ssh_tunnel/h200_pod_bash.sh}"
H200_POD_EXEC="${H200_POD_EXEC:-${H200_ACCESS_ROOT}/ssh_tunnel/h200_pod_exec.sh}"

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${CONTROLLER_DIR}/../.." && pwd)}"

# The GPFS project root. Overridable so no site layout is hard-wired into a
# committed file beyond the default the environment script already carries.
GPFS_PROJECT_ROOT="${GPFS_PROJECT_ROOT:-}"

STAGE=""
LABEL=""
GPU="0"
RUN_ID=""
SNAPSHOT_DIR=""
POLL_SECONDS="${POLL_SECONDS:-120}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-57600}"
STAGE_ARGS=()

usage() {
  sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --snapshot-dir) SNAPSHOT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; STAGE_ARGS=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

: "${H200_POD:?H200_POD must be exported by the caller}"
[ -n "${STAGE}" ] || { echo "--stage is required" >&2; exit 2; }
[ -n "${LABEL}" ] || { echo "--label is required" >&2; exit 2; }
[ -f "${REPO_ROOT}/scripts/transfer/${STAGE}" ] || {
  echo "no such stage: scripts/transfer/${STAGE}" >&2; exit 2; }

log() { printf '[external-baseline] %s %s\n' "$(date -Is)" "$*"; }

# ------------------------------------------------------------------- freeze

if [ -n "${RUN_ID}" ] || [ -n "${SNAPSHOT_DIR}" ]; then
  # Reuse. Both or neither -- one alone would silently write this condition's
  # results beside another run's snapshot, or run this snapshot's code into
  # another run's directory.
  [ -n "${RUN_ID}" ] && [ -n "${SNAPSHOT_DIR}" ] || {
    echo "--run-id and --snapshot-dir must be given together" >&2; exit 3; }
  # A snapshot that is not on the pod is the failure this option could otherwise
  # hide: the launch would start, find no stage file, and the poll loop would
  # report ABSENT as though the measurement had failed.
  "${H200_POD_BASH}" "test -f '${SNAPSHOT_DIR}/scripts/transfer/${STAGE}' && echo FOUND" \
      2>/dev/null | grep -q FOUND || {
    echo "reused snapshot ${SNAPSHOT_DIR} does not carry ${STAGE} on the pod; refusing" >&2
    exit 3
  }
  log "reusing snapshot run_id=${RUN_ID} (no freeze, no relay push)"
else
  log "freezing and pushing the code snapshot via the controller"
  FREEZE_OUT="$(cd "${REPO_ROOT}" && bash scripts/transfer/run_transfer_h200.sh --freeze-only)"
  RUN_ID="$(printf '%s\n' "${FREEZE_OUT}" | sed -n 's/^RUN_ID=//p')"
  SNAPSHOT_DIR="$(printf '%s\n' "${FREEZE_OUT}" | sed -n 's/^SNAPSHOT_DIR=//p')"
  [ -n "${RUN_ID}" ] && [ -n "${SNAPSHOT_DIR}" ] || {
    echo "the controller did not report a run id and snapshot dir; refusing" >&2
    printf '%s\n' "${FREEZE_OUT}" >&2
    exit 3
  }
  log "run_id=${RUN_ID}"
fi

if [ -z "${GPFS_PROJECT_ROOT}" ]; then
  # Derive it from the snapshot path the controller just reported rather than
  # restating a site layout: SNAPSHOT_DIR is <project-root>/packages/<run-id>.
  GPFS_PROJECT_ROOT="$(dirname "$(dirname "${SNAPSHOT_DIR}")")"
fi
OUT_DIR="${GPFS_PROJECT_ROOT}/results/external_baseline/${RUN_ID}/${LABEL}"
POD_LOG="${GPFS_PROJECT_ROOT}/logs/external_baseline/${RUN_ID}_${LABEL}.log"

# ------------------------------------------------------------------ dispatch

# The record EXP-R2-132 does not have. Written before the launch, so a run that
# dies still leaves its own dispatch behind.
LOCAL_RECORD="${REPO_ROOT}/logs/external_baseline/${RUN_ID}_${LABEL}.dispatch"
mkdir -p "$(dirname "${LOCAL_RECORD}")"
{
  printf 'run_id\t%s\n' "${RUN_ID}"
  printf 'snapshot\t%s\n' "${SNAPSHOT_DIR}"
  printf 'stage\t%s\n' "${STAGE}"
  printf 'label\t%s\n' "${LABEL}"
  printf 'gpu\t%s\n' "${GPU}"
  printf 'out\t%s\n' "${OUT_DIR}"
  printf 'stage_args\t%s\n' "${STAGE_ARGS[*]-}"
  printf 'dispatched_utc\t%s\n' "$(date -u -Is)"
} > "${LOCAL_RECORD}"
log "dispatch recorded at ${LOCAL_RECORD}"

# Detach INSIDE the pod. A foreground kubectl exec dies with the tunnel and
# takes the measurement with it; the campaign worker normally provides this and
# an external-baseline stage has no worker.
log "launching ${STAGE} on cuda:${GPU}"
"${H200_POD_EXEC}" -- bash -lc "
  set -euo pipefail
  export TRANSFER_PACKAGE_ROOT='${SNAPSHOT_DIR}'
  source '${SNAPSHOT_DIR}/scripts/transfer/h200_env.sh'
  : \"\${TRANSFER_PROGEN3_DIR:?must be exported by h200_env.sh or the caller}\"
  : \"\${TRANSFER_PROGEN3_SRC:?must be exported by h200_env.sh or the caller}\"
  mkdir -p '${OUT_DIR}' \"\$(dirname '${POD_LOG}')\"
  cd '${SNAPSHOT_DIR}'
  setsid nohup \"\${TRANSFER_PYTHON}\" '${SNAPSHOT_DIR}/scripts/transfer/${STAGE}' \
    --device cuda:${GPU} --out '${OUT_DIR}' ${STAGE_ARGS[*]-} \
    > '${POD_LOG}' 2>&1 < /dev/null &
  disown
  echo LAUNCHED
"

# --------------------------------------------------------------------- poll

# An item is absent only once the GPU it was scheduled on is observed idle.
# Checking for the artefact the moment the launcher returns reports MISSING on
# work that is still running -- the lesson logs/drivers/d2c_panel_h200.sh
# records, and the reason this loop is not a simple sleep.
#
# The idle test needs a startup grace period, and it earned one on its first
# run: 16_fitness_recovery.py spends minutes reading ProteinGym CSVs (one assay
# is 537k rows) before the model reaches the card, so an idle GPU during that
# window means "not started yet", not "finished without writing". The first
# invocation declared ABSENT after 0 s on a run that then completed normally.
GRACE_SECONDS="${GRACE_SECONDS:-600}"
sleep "${GRACE_SECONDS}"
waited="${GRACE_SECONDS}"
status="UNRESOLVED"
while [ "${waited}" -lt "${TIMEOUT_SECONDS}" ]; do
  if "${H200_POD_BASH}" "ls -1 '${OUT_DIR}' 2>/dev/null | grep -q '\.json$' && echo PRESENT" \
      2>/dev/null | grep -q PRESENT; then
    status="PRESENT"; break
  fi
  busy="$("${H200_POD_BASH}" \
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" 2>/dev/null || true)"
  if [ -n "${busy}" ] && ! printf '%s\n' "${busy}" \
      | awk -F', *' -v g="${GPU}" '$1==g && $2>1000{f=1}END{exit !f}'; then
    sleep 45
    if "${H200_POD_BASH}" "ls -1 '${OUT_DIR}' 2>/dev/null | grep -q '\.json$' && echo PRESENT" \
        2>/dev/null | grep -q PRESENT; then
      status="PRESENT"; break
    fi
    status="ABSENT"; break
  fi
  sleep "${POLL_SECONDS}"; waited=$((waited + POLL_SECONDS))
done
log "${LABEL} ${status} after ${waited}s"
[ "${status}" = "PRESENT" ] || exit 4

# --------------------------------------------------------- pull and verify

LOCAL_OUT="${REPO_ROOT}/results/transfer/external_baseline/${RUN_ID}/${LABEL}"
REMOTE_SUMS="$(mktemp)"
trap 'rm -f "${REMOTE_SUMS}"' EXIT
"${H200_POD_BASH}" "cd '${OUT_DIR}' && find . -type f -printf '%P\n' | sort | xargs sha256sum" \
  > "${REMOTE_SUMS}"
mkdir -p "$(dirname "${LOCAL_OUT}")"
"${H200_SYNC}" pull "${OUT_DIR}" "${LOCAL_OUT}"

# An external-baseline stage has no worker, so nothing writes a .manifests
# checksum for the pull to check. Admit a result only if the digests taken on
# each side agree; a silently truncated pull is a known failure mode here.
if ( cd "${LOCAL_OUT}" && sha256sum -c "${REMOTE_SUMS}" >/dev/null 2>&1 ); then
  log "digests verified; ${LOCAL_OUT} ADMITTED"
else
  echo "digest mismatch between pod and B; NOT ADMITTED: ${LOCAL_OUT}" >&2
  exit 5
fi
