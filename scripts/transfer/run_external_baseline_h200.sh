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
# The freeze is NOT reimplemented here, and neither is the pinning of the code
# it reads. run_transfer_h200.sh --freeze-only does both and prints RUN_ID,
# SNAPSHOT_DIR and, when pinned, PIN_COMMIT (Appendix B rule 12: one
# declaration, imported, never reimplemented).
#
# Privacy: no pod name is written here. Export H200_POD before invoking.
#
# Usage:
#   export H200_POD=<running-pod-name>
#   scripts/transfer/run_external_baseline_h200.sh \
#       --stage 16_fitness_recovery.py \
#       --label progenmech_stratified \
#       --gpu 0 \
#       --expect fitness_recovery.json \
#       -- --sampling progenmech_stratified --variants 1000
#
#   Repeat --gpu/--label pairs by invoking once per condition; each condition
#   gets its own results directory, because a resume key has no condition axis
#   and two conditions in one root would overwrite each other.
#
#   --expect <basename> is required. It names the artefact that means "done"
#   and must be an exact JSON basename: no directory, glob, regex, or traversal.
#   Any other JSON in the output directory is not completion. A stage that
#   READS an input staged into that same directory -- 20_retrieval_bound.py's
#   score stage reads wildtypes.json from --out -- must name its own artefact.
#
#   --timeout-seconds bounds the poll. The default is 86400 (24 h), enough for
#   a 12-24 h campaign; the hard ceiling is 172800 (48 h). TIMEOUT_SECONDS is
#   the same bound as an environment override.
#
#   To run several conditions of ONE comparison concurrently, freeze once at a
#   named commit and hand every invocation the same snapshot AND the same pin:
#
#     eval "$(scripts/transfer/run_transfer_h200.sh --pin HEAD --freeze-only)"
#     # sets RUN_ID, SNAPSHOT_DIR, PIN_COMMIT
#     scripts/transfer/run_external_baseline_h200.sh --pin "$PIN_COMMIT" \
#         --run-id "$RUN_ID" --snapshot-dir "$SNAPSHOT_DIR" \
#         --stage ... --label ... --gpu 0 --expect <basename.json> -- ... &
#
#   Two reasons to freeze once, one operational and one scientific. Four
#   controllers freezing at once collide on the shared relay's single temp
#   script path -- a hazard EXP-R2-122 recorded and a 20-second stagger did not
#   fix, because a push occupies the relay for minutes. And the arms of one
#   comparison must run the same code: a CLT and a PLT frozen from two snapshots
#   are not the controlled comparison they are reported as. Reusing a snapshot is
#   refused unless it is present on the pod, so this cannot silently run against
#   nothing.
#
#   --pin is what makes that reuse survive a working day. Without it, "the code"
#   means whatever is in the tree at the instant each command reads it, and
#   several agents commit into that tree: on 2026-08-12 the hash moved twice
#   inside one 4.5-minute freeze, five-plus dispatch attempts across three
#   campaigns were refused, and two campaigns were abandoned. The refusal is
#   correct and is unchanged -- a snapshot must never run under code it was not
#   frozen from -- so what --pin changes is the source being compared, from a
#   directory anyone may write to into a commit nobody can rewrite. Everything
#   below then reads that commit: the stage-file check, the code hash, and the
#   freeze itself. Uncommitted work is not dispatched under --pin; commit it
#   first, or leave the flag off and accept that the tree may move.
#
#   Dispatch each cell as its OWN command, never as one backgrounded && list:
#
#     export H200_POD=<pod>
#     scripts/transfer/run_external_baseline_h200.sh ... --gpu 0 -- ... &
#     scripts/transfer/run_external_baseline_h200.sh ... --gpu 1 -- ... &
#
#   `export H200_POD=... && cell0 & cell1 & cell2 &` backgrounds only the first
#   list, so the export stays inside that subshell and every later cell dies on
#   the unset variable. Two agents five hours apart lost three cells and then
#   eleven to exactly this; the driver refuses such a cell immediately and says
#   why, but a backgrounded refusal is easy to miss, so the shape to avoid is
#   written here, where a launcher looks, rather than only in the experiment log,
#   where the second agent did not look.

if [[ -n "${H200_ACCESS_ROOT:-}" ]]; then
  echo "H200_ACCESS_ROOT is no longer used; unset it and use HANGZHOU_COMPUTE_ROOT" >&2
  exit 2
fi
HANGZHOU_COMPUTE_ROOT="${HANGZHOU_COMPUTE_ROOT:-${HOME}/hangzhou-compute}"
H200_CLI="${H200_CLI:-${HANGZHOU_COMPUTE_ROOT}/h200}"

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${CONTROLLER_DIR}/../.." && pwd)}"
# shellcheck source=/dev/null
source "${CONTROLLER_DIR}/h200_orchestration.sh"

# Where this driver's own local output goes: the dispatch record it writes
# before launching, and the directory it pulls an admitted result into.
#
# Separate from REPO_ROOT, which answers a different question. REPO_ROOT is
# where the *code* is: it is what the stage-file check reads and what the
# code-hash comparison is computed over, so it must be the real checkout and a
# test cannot move it. Output location is not that question, and while one
# variable answered both, exercising the dispatch path wrote a `.dispatch`
# record for a run that never happened into the operational log directory an
# operator reads to see what was actually launched -- twice, once on
# 2026-08-07 and once on 2026-08-09, each naming a pytest temporary directory
# as its snapshot. That is Appendix B rule 29's failure ("a smoke run must not
# be written into the results tree") reaching the dispatch ledger, and it is
# fixed here by giving the two roles two names rather than by asking callers
# to remember.
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${REPO_ROOT}}"

# The GPFS project root. Overridable so no site layout is hard-wired into a
# committed file beyond the default the environment script already carries.
GPFS_PROJECT_ROOT="${GPFS_PROJECT_ROOT:-}"

STAGE=""
LABEL=""
GPU="0"
RUN_ID=""
SNAPSHOT_DIR=""
EXPECT=""
# --pin <commit-ish>, resolved to a full commit id below. Empty means the code
# is read from the working tree, which is the behaviour this driver has always
# had and is still what a caller who passes nothing gets.
PIN_REF=""
PIN_COMMIT=""
POLL_SECONDS="${POLL_SECONDS:-120}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-${TRANSFER_DEFAULT_TIMEOUT_SECONDS}}"
STAGE_ARGS=()

# The header from `Usage:` to the end of the leading comment block, which is
# where this file's operating instructions are. It used to be a fixed line
# range, which silently drifts off the end of what it was pointing at the first
# time the header grows.
usage() {
  sed -n '/^# Usage:/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --snapshot-dir) SNAPSHOT_DIR="$2"; shift 2 ;;
    --expect) EXPECT="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --pin) PIN_REF="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; STAGE_ARGS=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

: "${H200_POD:?H200_POD must be exported by the caller}"
[ -n "${STAGE}" ] || { echo "--stage is required" >&2; exit 2; }
[ -n "${LABEL}" ] || { echo "--label is required" >&2; exit 2; }
assert_timeout_seconds "${TIMEOUT_SECONDS}" || exit 2
[ -n "${EXPECT}" ] || { echo "--expect is required" >&2; exit 2; }
assert_expect_basename "${EXPECT}" || exit 2

log() { printf '[external-baseline] %s %s\n' "$(date -Is)" "$*"; }

# ------------------------------------------------------------- code source

# Resolve --pin once, here, and let every later code read follow it: the stage
# check immediately below, the code hash the reuse guard compares, and the
# freeze the controller performs. One dispatch therefore reads one code state,
# and a campaign whose cells all name the same commit reads the same one all
# day, whatever the working tree does meanwhile.
#
# CONTROLLER_PIN is passed on rather than re-implemented: the controller checks
# the commit out into a temporary worktree it owns and removes (see
# establish_pinned_code_root there), and this driver never materialises a tree
# of its own -- the one question it has to answer locally, whether the stage
# file exists at that commit, git answers out of the object store without a
# checkout.
CONTROLLER_PIN=()
if [ -n "${PIN_REF}" ]; then
  command -v git >/dev/null 2>&1 || {
    echo "--pin needs git(1) on PATH" >&2; exit 2; }
  PIN_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify --quiet "${PIN_REF}^{commit}")" || {
    echo "--pin ${PIN_REF} does not name a commit in ${REPO_ROOT}" >&2; exit 2; }
  CONTROLLER_PIN=(--pin "${PIN_COMMIT}")
  git -C "${REPO_ROOT}" cat-file -e "${PIN_COMMIT}:scripts/transfer/${STAGE}" 2>/dev/null || {
    echo "no such stage at ${PIN_COMMIT:0:12}: scripts/transfer/${STAGE}" >&2
    echo "(it may exist in the working tree; --pin dispatches the commit, not the tree)" >&2
    exit 2; }
else
  [ -f "${REPO_ROOT}/scripts/transfer/${STAGE}" ] || {
    echo "no such stage: scripts/transfer/${STAGE}" >&2; exit 2; }
fi

# ------------------------------------------------------------------- freeze

if [ -n "${RUN_ID}" ] || [ -n "${SNAPSHOT_DIR}" ]; then
  # Reuse. Both or neither -- one alone would silently write this condition's
  # results beside another run's snapshot, or run this snapshot's code into
  # another run's directory.
  [ -n "${RUN_ID}" ] && [ -n "${SNAPSHOT_DIR}" ] || {
    echo "--run-id and --snapshot-dir must be given together" >&2; exit 3; }
  # The run-id's trailing segment must be the hash of the code this dispatch
  # reads -- the pinned commit under --pin, the working tree otherwise. This is
  # resolve_run_id's rule, applied at the one other place a snapshot can be
  # adopted, and asked of the controller rather than reimplemented (Appendix B
  # rule 12). Without it this option silently runs stale code: --replacement-kind
  # was added to a stage after its snapshot was frozen, four launches died on
  # `unrecognized arguments`, and each was reported LAUNCHED and then polled for
  # ten minutes.
  #
  # The controller's own diagnostics are kept and shown on failure. They used to
  # be discarded, so a controller that refused for its own reason -- an
  # unresolvable pin, a missing file -- arrived here as the bare and misleading
  # "could not compute the current code hash".
  HASH_OUT="$(cd "${REPO_ROOT}" && bash scripts/transfer/run_transfer_h200.sh \
      ${CONTROLLER_PIN[@]+"${CONTROLLER_PIN[@]}"} --print-code-hash 2>&1)" || {
    printf '%s\n' "${HASH_OUT}" >&2
    echo "the controller could not hash the code to compare against; refusing" >&2
    exit 3
  }
  CURRENT_HASH="$(printf '%s\n' "${HASH_OUT}" | sed -n 's/^CODE_HASH=//p')"
  [ -n "${CURRENT_HASH}" ] || {
    printf '%s\n' "${HASH_OUT}" >&2
    echo "could not compute the current code hash" >&2
    exit 3
  }
  case "${RUN_ID}" in
    *_"${CURRENT_HASH:0:12}") ;;
    *)
      echo "refusing to reuse ${RUN_ID}: it was minted from different code than" >&2
      if [ -n "${PIN_COMMIT}" ]; then
        echo "commit ${PIN_COMMIT:0:12} carries (its hash is ${CURRENT_HASH:0:12})." >&2
        echo "Pin the commit the snapshot was frozen from, or freeze a new snapshot;" >&2
      else
        echo "is on disk now (current hash ${CURRENT_HASH:0:12}). Freeze a new snapshot;" >&2
        echo "if the tree moved under another agent's commit rather than under your own" >&2
        echo "edit, pass --pin <commit> so this reads a commit instead of the tree;" >&2
      fi
      echo "a reused snapshot runs the code it was frozen with, not the code you edited." >&2
      exit 3
      ;;
  esac
  # A snapshot that is not on the pod is the other failure this option could
  # hide: the launch would start, find no stage file, and the poll loop would
  # report ABSENT as though the measurement had failed.
  "${H200_CLI}" bash "test -f '${SNAPSHOT_DIR}/scripts/transfer/${STAGE}' && echo FOUND" \
      2>/dev/null | grep -q FOUND || {
    echo "reused snapshot ${SNAPSHOT_DIR} does not carry ${STAGE} on the pod; refusing" >&2
    exit 3
  }
  if [ -n "${PIN_COMMIT}" ]; then
    log "reusing snapshot run_id=${RUN_ID} (verified against commit ${PIN_COMMIT:0:12}; no relay push)"
  else
    log "reusing snapshot run_id=${RUN_ID} (verified against the code on disk; no relay push)"
  fi
else
  log "freezing and pushing the code snapshot via the controller"
  FREEZE_OUT="$(cd "${REPO_ROOT}" && bash scripts/transfer/run_transfer_h200.sh \
      ${CONTROLLER_PIN[@]+"${CONTROLLER_PIN[@]}"} --freeze-only)"
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
LOCAL_RECORD="${LOCAL_OUTPUT_ROOT}/logs/external_baseline/${RUN_ID}_${LABEL}.dispatch"
mkdir -p "$(dirname "${LOCAL_RECORD}")"
{
  printf 'run_id\t%s\n' "${RUN_ID}"
  printf 'snapshot\t%s\n' "${SNAPSHOT_DIR}"
  # Which code state this cell was dispatched against, in the ledger an operator
  # reads. A campaign's cells must all show the same one.
  printf 'code_source\t%s\n' "${PIN_COMMIT:-working tree ${REPO_ROOT}}"
  printf 'stage\t%s\n' "${STAGE}"
  printf 'label\t%s\n' "${LABEL}"
  printf 'gpu\t%s\n' "${GPU}"
  printf 'out\t%s\n' "${OUT_DIR}"
  printf 'expect\t%s\n' "${EXPECT}"
  printf 'stage_args\t%s\n' "${STAGE_ARGS[*]-}"
  printf 'dispatched_utc\t%s\n' "$(date -u -Is)"
} > "${LOCAL_RECORD}"
log "dispatch recorded at ${LOCAL_RECORD}"

# Detach INSIDE the pod. A foreground kubectl exec dies with the tunnel and
# takes the measurement with it; the campaign worker normally provides this and
# an external-baseline stage has no worker. The wrapper records host state
# before and after the stage, including on failure or interruption.
HOST_PRE="${GPFS_PROJECT_ROOT}/logs/external_baseline/${RUN_ID}_${LABEL}.host_pre.txt"
HOST_POST="${GPFS_PROJECT_ROOT}/logs/external_baseline/${RUN_ID}_${LABEL}.host_post.txt"
WRAP="${GPFS_PROJECT_ROOT}/logs/external_baseline/${RUN_ID}_${LABEL}.wrap.sh"
log "launching ${STAGE} on cuda:${GPU}"
"${H200_CLI}" exec -- bash -lc "
  set -euo pipefail
  export TRANSFER_PACKAGE_ROOT='${SNAPSHOT_DIR}'
  source '${SNAPSHOT_DIR}/scripts/transfer/h200_env.sh'
  : \"\${TRANSFER_PROGEN3_DIR:?must be exported by h200_env.sh or the caller}\"
  : \"\${TRANSFER_PROGEN3_SRC:?must be exported by h200_env.sh or the caller}\"
  mkdir -p '${OUT_DIR}' \"\$(dirname '${POD_LOG}')\" \"\$(dirname '${WRAP}')\"
  cat > '${WRAP}' <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
source \"\${TRANSFER_PACKAGE_ROOT}/scripts/transfer/h200_env.sh\"
source \"\${TRANSFER_PACKAGE_ROOT}/scripts/transfer/h200_orchestration.sh\"
run_wrapped_external_stage \"\$@\"
WRAP
  chmod +x '${WRAP}'
  export XFER_STAGE='${SNAPSHOT_DIR}/scripts/transfer/${STAGE}'
  export XFER_OUT='${OUT_DIR}'
  export XFER_GPU='${GPU}'
  export XFER_HOST_PRE='${HOST_PRE}'
  export XFER_HOST_POST='${HOST_POST}'
  cd '${SNAPSHOT_DIR}'
  setsid nohup bash '${WRAP}' ${STAGE_ARGS[*]-} \
    > '${POD_LOG}' 2>&1 < /dev/null &
  disown
  echo LAUNCHED
"

# ------------------------------------------------------------- liveness check

# A stage that dies in its first seconds -- a bad flag, a missing checkpoint, an
# import error -- printed LAUNCHED and was then polled for the full grace period
# before the idle-GPU test called it ABSENT. ABSENT is the verdict for "ran and
# wrote nothing", which is a measurement outcome; this is a dispatch failure and
# must not be reported as one. Checked after a short settle rather than by
# tracking a pid, because the access layer returns 0 whatever the remote command
# did (L20), so a sentinel read out of the stage's own log is the only signal
# that can say no.
sleep "${LIVENESS_SETTLE_SECONDS:-45}"
EARLY="$("${H200_CLI}" bash \
  "tail -n 40 '${POD_LOG}' 2>/dev/null | grep -c -E 'Traceback|error: unrecognized arguments|error: argument|No such file or directory|ModuleNotFoundError|CUDA out of memory' || true" \
  2>/dev/null | tr -dc '0-9')"
if [ -n "${EARLY}" ] && [ "${EARLY}" -gt 0 ]; then
  log "${LABEL} DIED AT DISPATCH"
  "${H200_CLI}" bash "tail -n 20 '${POD_LOG}'" 2>/dev/null >&2 || true
  echo "the stage exited during start-up; this is a dispatch failure, not an ABSENT" >&2
  echo "measurement. Nothing was scheduled on cuda:${GPU}." >&2
  exit 6
fi

# --------------------------------------------------------------------- poll

# What "done" looks like in the output directory. Declared once and asked twice
# below, because the two call sites must agree: a poll that accepted a file the
# confirming re-poll rejected would turn a finished run into an ABSENT.
present() {
  "${H200_CLI}" bash \
    "test -f '${OUT_DIR}/${EXPECT}' && echo PRESENT" \
    2>/dev/null | grep -q PRESENT
}

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
  if present; then
    status="PRESENT"; break
  fi
  busy="$("${H200_CLI}" bash \
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" 2>/dev/null || true)"
  if [ -n "${busy}" ] && ! printf '%s\n' "${busy}" \
      | awk -F', *' -v g="${GPU}" '$1==g && $2>1000{f=1}END{exit !f}'; then
    sleep 45
    if present; then
      status="PRESENT"; break
    fi
    status="ABSENT"; break
  fi
  sleep "${POLL_SECONDS}"; waited=$((waited + POLL_SECONDS))
done
log "${LABEL} ${status} after ${waited}s"
[ "${status}" = "PRESENT" ] || exit 4

ADMIT_PATH="${OUT_DIR}/${EXPECT}"
ADMIT_DIGEST="$("${H200_CLI}" bash "
  set -euo pipefail
  python3 -c 'import json,sys
path=sys.argv[1]
text=open(path,encoding=\"utf-8\").read()
if not text.strip():
    raise SystemExit(1)
json.loads(text)' '${ADMIT_PATH}'
  sha256sum '${ADMIT_PATH}'
" 2>/dev/null | awk '{print $1}')" || ADMIT_DIGEST=""
if [ -z "${ADMIT_DIGEST}" ]; then
  echo "expected artefact is missing, empty, or not valid JSON: ${ADMIT_PATH}" >&2
  exit 4
fi
"${H200_CLI}" bash "printf '%s  %s\\n' '${ADMIT_DIGEST}' '$(basename "${ADMIT_PATH}")' > '${ADMIT_PATH}.sha256.tmp' && mv -f '${ADMIT_PATH}.sha256.tmp' '${ADMIT_PATH}.sha256'" >/dev/null
log "admitted digest ${ADMIT_DIGEST} for $(basename "${ADMIT_PATH}")"

# --------------------------------------------------------- pull and verify

LOCAL_OUT="${LOCAL_OUTPUT_ROOT}/results/transfer/external_baseline/${RUN_ID}/${LABEL}"
REMOTE_SUMS="$(mktemp)"
trap 'rm -f "${REMOTE_SUMS}"' EXIT
"${H200_CLI}" bash "cd '${OUT_DIR}' && find . -type f -printf '%P\n' | sort | xargs sha256sum" \
  > "${REMOTE_SUMS}"
mkdir -p "$(dirname "${LOCAL_OUT}")"
"${H200_CLI}" sync pull "${OUT_DIR}" "${LOCAL_OUT}"

# An external-baseline stage has no worker, so nothing writes a .manifests
# checksum for the pull to check. Admit a result only if the digests taken on
# each side agree; a silently truncated pull is a known failure mode here.
if ( cd "${LOCAL_OUT}" && sha256sum -c "${REMOTE_SUMS}" >/dev/null 2>&1 ); then
  log "digests verified; ${LOCAL_OUT} ADMITTED"
else
  echo "digest mismatch between pod and B; NOT ADMITTED: ${LOCAL_OUT}" >&2
  exit 5
fi
