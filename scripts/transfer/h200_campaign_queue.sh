#!/usr/bin/env bash
set -euo pipefail

# ############################################################################
# VALIDATED (2026-08-17). Steps 1 to 5 of "First run" below have all been
# executed in a real pod and all passed: it parsed and resolved a manifest,
# launched a real stage as its own child on a real card, read that child's exit
# status from wait(2), wrote and atomically rewrote the status file through
# every transition, took and released the lock, crossed the slot barrier, and on
# an identical re-dispatch reported skipped-complete without launching anything.
# The artefact its cell produced is bit-identical, on every scientific field, to
# the one the single-cell driver produced from the same code and arguments.
#
# Two limits that remain. It has never run a campaign of more than one cell or
# more than one slot, so concurrency within a slot and the barrier between two
# populated slots are exercised only by construction. And a cell that FAILS has
# never been observed: exited-nonzero, exited-ok-no-artifact and
# refused-busy-gpu are all unexecuted paths.
# ############################################################################
#
# An IN-POD sequential campaign runner: one dispatch, a whole campaign.
#
# The problem it solves. run_external_baseline_h200.sh dispatches ONE cell per
# invocation and then polls it from the workstation, so an N-round campaign
# needs N successful dispatches and N successful polls across the transit link.
# When that link is unstable, every extra crossing is another chance to lose a
# round. This runner is launched ONCE, detached in the pod exactly the way that
# driver detaches a stage, and from then on the campaign needs no dispatch at
# all: the observation channel is a single small status file on GPFS that one
# `cat` retrieves (see "The status file" below).
#
# What it does NOT change. The dispatch path per cell is the driver's, verbatim
# -- same interpreter, same environment sourcing, same output and log paths.
# This file adds sequencing and a status file; it invents no convention. The
# reuse points are quoted against run_external_baseline_h200.sh in "Reuse"
# below so a reviewer can diff them by eye.
#
# What it deliberately does not do. It does not pull results, does not freeze
# or push a snapshot, and does not talk to the workstation. Freezing stays with
# run_transfer_h200.sh --freeze-only (Appendix B rule 12: one declaration,
# imported, never reimplemented) and retrieval stays with
# pull_records_h200.sh.
#
# ---------------------------------------------------------------- Usage
#
# On the workstation, once, after freezing every snapshot the campaign needs.
# Freeze SEQUENTIALLY: concurrent freezes collide on the shared relay's single
# temp script path, which run_external_baseline_h200.sh's header records.
#
#   export H200_POD=<running-pod-name>
#   eval "$(scripts/transfer/run_transfer_h200.sh --pin <commit-cc> --freeze-only)"
#   CC_SNAPSHOT="$SNAPSHOT_DIR"
#   eval "$(scripts/transfer/run_transfer_h200.sh --pin <commit-r207> --freeze-only)"
#   R207_SNAPSHOT="$SNAPSHOT_DIR"
#
#   ~/hangzhou-remote/ssh_tunnel/h200_pod_exec.sh -- bash -lc "
#     setsid nohup bash '${CC_SNAPSHOT}/scripts/transfer/h200_campaign_queue.sh' \
#       --manifest '${CC_SNAPSHOT}/scripts/transfer/<campaign>.tsv' \
#       --snapshot cc='${CC_SNAPSHOT}' \
#       --snapshot r207='${R207_SNAPSHOT}' \
#       > /dev/null 2>&1 < /dev/null &
#     disown
#     echo QUEUE_LAUNCHED
#   "
#
# A campaign that fits one pin passes one --snapshot and names one key; the two
# above are the general form, not a requirement.
#
# The manifest reaches the pod only INSIDE a snapshot, so revising a manifest
# means freezing again. That is the intended cost rather than an oversight: the
# manifest is part of the run's identity in the same way this launcher is, and
# a manifest written straight onto GPFS would be a campaign definition nobody
# can trace to a commit. Budget one freeze per manifest revision.
#
# `echo QUEUE_LAUNCHED` is not evidence the runner is running -- the access
# layer returns 0 whatever the remote command did. The evidence is the status
# file appearing, which it does before the first cell is launched.
#
# Then, however rarely the link allows:
#
#   ~/hangzhou-remote/ssh_tunnel/h200_pod_bash.sh \
#     "cat <gpfs>/logs/external_baseline/<campaign>.status.tsv"
#
# A campaign may need more than one snapshot because its cells are pinned to
# different commits. Hence --snapshot <key>=<dir>, repeated, and a `key` column
# in the manifest. The runner's OWN copy comes from whichever snapshot launched
# it and is independent of any cell's code provenance: it never imports
# repository python and only ever invokes each cell's own snapshot entry point.
#
# The worked case, kept here as the reference because it is the one that showed
# a single freeze cannot serve every campaign. EXP-R2-206's Crosscoder cells
# need a commit carrying 32_crosscoder.py; EXP-R2-207's trainer cells are pinned
# to bd6ff99, which predates that file (`git cat-file -e
# bd6ff99:scripts/transfer/32_crosscoder.py` fails) and which carries
# 17_train_transcoder.py and src/transfer/transcoders.py byte-identical to
# 04fdfa5, the code state that trained every baseline those cells are read
# against. One snapshot for both would either run the Crosscoder against a
# commit that has no Crosscoder, or run the trainer cells against a trainer that
# has moved -- HEAD differed from bd6ff99 by 316 insertions and 95 deletions
# across those two files. So: freeze sequentially, once per pin, and pass every
# resulting directory to one dispatch.
#
# ---------------------------------------------------------------- Manifest
#
# Tab-separated, `#` comments and blank lines ignored, one cell per line:
#
#   slot <TAB> key <TAB> gpu <TAB> stage <TAB> label <TAB> env <TAB> args
#
#   slot   integer. Cells sharing a slot run concurrently; slot N+1 starts only
#          when every cell of slot N has exited. Ascending numeric order.
#   key    which --snapshot this cell's code comes from.
#   gpu    card index. Unique within a slot -- two cells on one card is refused
#          at parse time, not discovered at OOM.
#   stage  file name under <snapshot>/scripts/transfer/.
#   env    space-separated KEY=VALUE applied after h200_env.sh (so a cell can
#          override it), or `-` for none.
#   args   the stage's arguments, whitespace separated. NO argument may contain
#          whitespace. `--device` and `--out` must NOT appear: this runner
#          injects both, and a second spelling of either is a second
#          declaration of where the run went.
#
# The literal token ${TRANSFER_MODEL_BASE_DIR} in `args` is substituted in-pod
# after h200_env.sh is sourced. That keeps the model root declared once, in
# h200_env.sh, rather than copied into a manifest (Single-Source Principle).
# No other expansion happens: the args field is not eval'd.
#
# ------------------------------------------------------------ Status file
#
# Rewritten atomically (temp file + mv) on every state change, at
# <gpfs>/logs/external_baseline/<campaign>.status.tsv unless --status says
# otherwise. A `#` metadata block, then one TSV row per cell:
#
#   slot label gpu state exit started_utc ended_utc artifact log
#
# States: pending, running, exited-ok, exited-ok-no-artifact, exited-nonzero,
# skipped-complete, refused-busy-gpu. `exited-ok-no-artifact` is the driver's
# ABSENT -- ran and wrote nothing, a measurement outcome and not a failure.
# `exited-nonzero` carries the real exit code, which this runner has and the
# driver does not: a cell here is this process's own child, so its status is
# read from wait(2) rather than inferred from a sentinel grep of its log.
#
# The metadata block always carries `# FAILURES` and `# NO-RECORD` lines, both
# present and reading 0 when there are none, so
# `grep -E '^# (FAILURES|NO-RECORD)' <status>` is a complete answer to "did
# anything break", and their absence means the file is truncated rather than
# clean. They are two lines and not one because the repository's state
# vocabulary distinguishes them: a nonzero exit is a failure, a cell that ran
# and wrote nothing is a measurement outcome. Both need an operator; only one
# is a defect.
#
# ---------------------------------------------------------- Resumability
#
# A cell is skipped iff a `*.json` already exists in its own output directory.
# That is the same completion test run_external_baseline_h200.sh applies by
# default (its PRESENT_PATTERN), and it is sound for these stages because both
# write their record LAST and write it atomically: 17_train_transcoder.py does
# `torch.save(... .pt)` then `write_json(... .json)`, and src.transfer.io's
# write_json is documented "Write sorted, indented JSON atomically". So a JSON
# implies a finished `.pt` beside it, and an interrupted cell leaves a partial
# `.pt` with no JSON and is re-run, which rewrites that `.pt` whole.
#
# Its limits, stated because they are real:
#   * A stage that STAGES AN INPUT into its own --out directory would read as
#     complete on that input (this is why the driver has --expect). Neither
#     stage in this campaign does; a manifest that adds one needs an expect
#     column first.
#   * The test cannot tell a finished cell from one still running under an
#     ORPHANED earlier runner. The lock below catches the common case and the
#     per-cell idle-card check catches the rest; neither is a proof.
#   * It does not verify the record's content, only that one exists.
#
# ---------------------------------------------------------------- Reuse
#
# Quoted from run_external_baseline_h200.sh so the match is checkable:
#
#   OUT_DIR="${GPFS_PROJECT_ROOT}/results/external_baseline/${RUN_ID}/${LABEL}"
#   POD_LOG="${GPFS_PROJECT_ROOT}/logs/external_baseline/${RUN_ID}_${LABEL}.log"
#
#   GPFS_PROJECT_ROOT="$(dirname "$(dirname "${SNAPSHOT_DIR}")")"
#
#   export TRANSFER_PACKAGE_ROOT='${SNAPSHOT_DIR}'
#   source '${SNAPSHOT_DIR}/scripts/transfer/h200_env.sh'
#   : "${TRANSFER_PROGEN3_DIR:?must be exported by h200_env.sh or the caller}"
#   : "${TRANSFER_PROGEN3_SRC:?must be exported by h200_env.sh or the caller}"
#   mkdir -p '${OUT_DIR}' "$(dirname '${POD_LOG}')"
#   cd '${SNAPSHOT_DIR}'
#   setsid nohup "${TRANSFER_PYTHON}" '${SNAPSHOT_DIR}/scripts/transfer/${STAGE}' \
#     --device cuda:${GPU} --out '${OUT_DIR}' ${STAGE_ARGS[*]-} \
#     > '${POD_LOG}' 2>&1 < /dev/null &
#
# Three deliberate differences, each with its reason:
#   1. No `setsid nohup` per cell. A cell must be a child this runner can wait
#      on -- that is what makes the round barrier a barrier and what makes the
#      exit code real. Detachment moves up one level: the RUNNER is what gets
#      `setsid nohup`'d, once, by the dispatch above, which is the same proven
#      detachment applied at the level that now needs it.
#   2. The pod's login profile is sourced once for the runner rather than once
#      per cell (the driver runs each cell under `bash -lc`). h200_env.sh is
#      still sourced per cell, in a subshell, so two snapshots with different
#      environment files do not leak into each other.
#   3. The idle-card threshold reused from the driver's poll loop is applied
#      BEFORE launch as well as after: `$2>1000` MiB on
#      `nvidia-smi --query-gpu=index,memory.used`.
#
# ------------------------------------------------------------- First run
#
# Ordered so the cheapest disconfirming check comes first:
#   1. `bash -n` this file and the manifest's stage list by eye.  PASSED 08-17.
#   2. On the workstation, --dry-run: it parses and validates the manifest,
#      prints every resolved command, and touches nothing.        PASSED 08-17.
#   3. In the pod, run --dry-run against the real snapshots: this is the first
#      check that the snapshot paths, stage files and GPFS roots are real.
#                                                                 PASSED 08-17.
#   4. Run a one-cell one-slot throwaway manifest whose stage is cheap, and
#      confirm the status file appears, transitions, and ends `exited-ok`.
#                                                                 PASSED 08-17.
#   5. Re-run step 4 unchanged and confirm the cell reports skipped-complete.
#                                                                 PASSED 08-17.
#   6. Only then dispatch the campaign.
#
# What steps 4 and 5 returned, recorded so the claim is checkable. The cell was
# EXP-R2-202's own base/protein spectrum cell run from the snapshot frozen at its
# own pin, under a label with no directory on GPFS. Step 4: launched on cuda:0 as
# a child pid, exited 0 in three minutes, status file went pending -> running ->
# exited-ok with the artefact path recorded, `# FAILURES 0` and `# NO-RECORD 0`
# throughout, slot barrier reached, lock released after the settle. Step 5, the
# identical manifest re-dispatched: `skipped-complete`, nothing launched, the
# runner exited 0. And the artefact compares bit-identical to the driver's on
# `spectrum`, `verdict`, `condition`, `controls` and `loader_gate`; the only
# fields that differ are the card index and two wall-clock timings.
#
# What step 3 confirmed in the pod, recorded so it is not re-litigated: bash
# 5.2.21 (`declare -A` available), `date -u`, `sha256sum`, `mktemp`, `setsid`,
# `nohup`, `flock`, `xargs -r` on empty input, `find -printf` and `grep -m 1`
# all present; four cards; and `nvidia-smi --query-gpu=index,memory.used
# --format=csv,noheader,nounits` emitting `0, 45491`, which is the shape
# gpu_is_busy's `awk -F', *'` parses. Steps 4 and 5 were deliberately skipped
# because all four cards were occupied and a throwaway cell would have had to
# contend with pre-registered work.
#
# One property step 3 does NOT check, and the manifest is where it bites: the
# resume rule keys on the output directory, which is the LABEL. A manifest whose
# labels differ from the labels its cells were actually dispatched under will
# re-run completed cells rather than skip them, and a dry run cannot see it
# because a dry run never consults the results tree. Diff the manifest's labels
# against `ls <gpfs>/results/external_baseline/<run-id>/` before dispatching.

usage() {
  sed -n '/^# ---------------------------------------------------------------- Usage/,/^# ---------------------------------------------------------------- Manifest/p' \
    "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

MANIFEST=""
STATUS=""
CAMPAIGN=""
DRY_RUN=0
POLL_SECONDS="${POLL_SECONDS:-30}"
SLOT_SETTLE_SECONDS="${SLOT_SETTLE_SECONDS:-60}"
BUSY_MIB="${BUSY_MIB:-1000}"
declare -A SNAPSHOT_FOR=()

while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --status) STATUS="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --slot-settle-seconds) SLOT_SETTLE_SECONDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --snapshot)
      case "$2" in
        *=*) SNAPSHOT_FOR["${2%%=*}"]="${2#*=}" ;;
        *) echo "--snapshot takes <key>=<directory>, got: $2" >&2; exit 2 ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "${MANIFEST}" ] || { echo "--manifest is required" >&2; exit 2; }
[ -f "${MANIFEST}" ] || { echo "no such manifest: ${MANIFEST}" >&2; exit 2; }
[ "${#SNAPSHOT_FOR[@]}" -gt 0 ] || { echo "at least one --snapshot <key>=<dir> is required" >&2; exit 2; }

now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[campaign-queue] %s %s\n' "$(now_utc)" "$*"; }

# ------------------------------------------------------------------ snapshots
#
# Every snapshot is checked here rather than at first use, so a mistyped path
# is refused before any cell starts instead of after the first round.

FIRST_SNAPSHOT=""
for key in "${!SNAPSHOT_FOR[@]}"; do
  dir="${SNAPSHOT_FOR[$key]}"
  [ -d "${dir}" ] || { echo "snapshot ${key}: no such directory: ${dir}" >&2; exit 2; }
  [ -f "${dir}/scripts/transfer/h200_env.sh" ] || {
    echo "snapshot ${key} does not look like a code snapshot: no scripts/transfer/h200_env.sh under ${dir}" >&2
    exit 2; }
  [ -n "${FIRST_SNAPSHOT}" ] || FIRST_SNAPSHOT="${dir}"
done

# SNAPSHOT_DIR is <project-root>/packages/<run-id>: the driver's own derivation,
# not a restated site layout.
GPFS_PROJECT_ROOT="$(dirname "$(dirname "${FIRST_SNAPSHOT}")")"
[ -n "${CAMPAIGN}" ] || { CAMPAIGN="$(basename "${MANIFEST}")"; CAMPAIGN="${CAMPAIGN%.*}"; }
[ -n "${STATUS}" ] || STATUS="${GPFS_PROJECT_ROOT}/logs/external_baseline/${CAMPAIGN}.status.tsv"
QUEUE_LOG="${GPFS_PROJECT_ROOT}/logs/external_baseline/${CAMPAIGN}.queue.log"

# ------------------------------------------------------------------- manifest

N=0
declare -a C_SLOT=() C_KEY=() C_GPU=() C_STAGE=() C_LABEL=() C_ENV=() C_ARGS=()
declare -a C_OUT=() C_LOG=() C_STATE=() C_EXIT=() C_START=() C_END=() C_ART=() C_PID=()

line_no=0
while IFS=$'\t' read -r slot key gpu stage label cellenv args || [ -n "${slot:-}" ]; do
  line_no=$((line_no + 1))
  case "${slot}" in ''|'#'*) continue ;; esac
  for field in "${key}" "${gpu}" "${stage}" "${label}" "${cellenv}" "${args}"; do
    [ -n "${field}" ] || {
      echo "manifest line ${line_no}: expected 7 tab-separated fields" >&2; exit 2; }
  done
  case "${slot}" in *[!0-9]*) echo "manifest line ${line_no}: slot must be an integer, got '${slot}'" >&2; exit 2 ;; esac
  case "${gpu}" in *[!0-9]*) echo "manifest line ${line_no}: gpu must be an integer, got '${gpu}'" >&2; exit 2 ;; esac
  [ -n "${SNAPSHOT_FOR[$key]+set}" ] || {
    echo "manifest line ${line_no}: no --snapshot was given for key '${key}'" >&2; exit 2; }
  snapshot="${SNAPSHOT_FOR[$key]}"
  [ -f "${snapshot}/scripts/transfer/${stage}" ] || {
    echo "manifest line ${line_no}: snapshot '${key}' does not carry scripts/transfer/${stage}" >&2
    echo "(a cell must run the code its own snapshot was frozen with)" >&2
    exit 2; }
  case " ${args} " in
    *" --device "*|*" --out "*)
      echo "manifest line ${line_no}: --device and --out are injected by this runner and must not appear in args" >&2
      exit 2 ;;
  esac

  run_id="$(basename "${snapshot}")"
  C_SLOT+=("${slot}"); C_KEY+=("${key}"); C_GPU+=("${gpu}")
  C_STAGE+=("${stage}"); C_LABEL+=("${label}"); C_ENV+=("${cellenv}"); C_ARGS+=("${args}")
  C_OUT+=("${GPFS_PROJECT_ROOT}/results/external_baseline/${run_id}/${label}")
  C_LOG+=("${GPFS_PROJECT_ROOT}/logs/external_baseline/${run_id}_${label}.log")
  C_STATE+=("pending"); C_EXIT+=("-"); C_START+=("-"); C_END+=("-"); C_ART+=("-"); C_PID+=("-")
  N=$((N + 1))
done < "${MANIFEST}"

[ "${N}" -gt 0 ] || { echo "manifest ${MANIFEST} declares no cells" >&2; exit 2; }

# Two collisions that are cheap to refuse and expensive to discover: one label
# twice would put two cells in one results directory (a resume key has no
# condition axis), and one card twice in one slot would run two cells on one
# card, which is how a pre-registered memory budget stops meaning anything.
for ((i = 0; i < N; i++)); do
  for ((j = i + 1; j < N; j++)); do
    [ "${C_LABEL[$i]}" != "${C_LABEL[$j]}" ] || {
      echo "duplicate label '${C_LABEL[$i]}': two cells would share one results directory" >&2; exit 2; }
    if [ "${C_SLOT[$i]}" = "${C_SLOT[$j]}" ] && [ "${C_GPU[$i]}" = "${C_GPU[$j]}" ]; then
      echo "slot ${C_SLOT[$i]} puts '${C_LABEL[$i]}' and '${C_LABEL[$j]}' both on cuda:${C_GPU[$i]}" >&2
      exit 2
    fi
  done
done

SLOTS="$(printf '%s\n' "${C_SLOT[@]}" | sort -n -u)"

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "campaign      ${CAMPAIGN}"
  echo "manifest      ${MANIFEST}"
  echo "status        ${STATUS}"
  echo "project root  ${GPFS_PROJECT_ROOT}"
  echo "cells         ${N} in $(printf '%s\n' "${SLOTS}" | wc -l) slot(s)"
  for slot in ${SLOTS}; do
    echo "--- slot ${slot}"
    for ((i = 0; i < N; i++)); do
      [ "${C_SLOT[$i]}" = "${slot}" ] || continue
      printf '  cuda:%s %s\n' "${C_GPU[$i]}" "${C_LABEL[$i]}"
      printf '    out  %s\n' "${C_OUT[$i]}"
      printf '    log  %s\n' "${C_LOG[$i]}"
      printf '    env  %s\n' "${C_ENV[$i]}"
      printf '    cmd  ${TRANSFER_PYTHON} %s/scripts/transfer/%s --device cuda:%s --out %s %s\n' \
        "${SNAPSHOT_FOR[${C_KEY[$i]}]}" "${C_STAGE[$i]}" "${C_GPU[$i]}" "${C_OUT[$i]}" "${C_ARGS[$i]}"
    done
  done
  exit 0
fi

# ----------------------------------------------------------------------- lock
#
# Two runners on one status file would interleave two rewrites of it and, worse,
# would each relaunch the other's running cells. PID reuse can in principle make
# this refuse when it should not; refusing is the safe direction and the message
# says what to check.

mkdir -p "$(dirname "${STATUS}")" "$(dirname "${QUEUE_LOG}")"
LOCK="${STATUS}.lock"
if [ -e "${LOCK}" ]; then
  other="$(cat "${LOCK}" 2>/dev/null || true)"
  if [ -n "${other}" ] && kill -0 "${other}" 2>/dev/null; then
    echo "a campaign queue runner (pid ${other}) is already live on ${STATUS}; refusing" >&2
    echo "if that pid is not a queue runner, remove ${LOCK} and re-dispatch" >&2
    exit 3
  fi
  log "stale lock from pid ${other:-unknown}; taking it over"
fi
printf '%s\n' "$$" > "${LOCK}"
trap 'rm -f "${LOCK}"' EXIT

exec >> "${QUEUE_LOG}" 2>&1
log "campaign ${CAMPAIGN}: ${N} cells, manifest ${MANIFEST}"

# --------------------------------------------------------------- status file

MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
STARTED_UTC="$(now_utc)"
CURRENT_SLOT="-"

write_status() {
  local tmp="${STATUS}.tmp.$$" i state
  local -A tally=()
  local failures="" n_failed=0 silent="" n_silent=0
  for state in pending running exited-ok exited-ok-no-artifact exited-nonzero \
               skipped-complete refused-busy-gpu; do
    tally["${state}"]=0
  done
  for ((i = 0; i < N; i++)); do
    state="${C_STATE[$i]}"
    tally["${state}"]=$(( ${tally["${state}"]:-0} + 1 ))
    case "${state}" in
      exited-nonzero|refused-busy-gpu)
        n_failed=$((n_failed + 1))
        failures="${failures}${failures:+, }${C_LABEL[$i]}(${state}:${C_EXIT[$i]})"
        ;;
      exited-ok-no-artifact)
        n_silent=$((n_silent + 1))
        silent="${silent}${silent:+, }${C_LABEL[$i]}"
        ;;
    esac
  done
  {
    printf '# campaign\t%s\n' "${CAMPAIGN}"
    printf '# manifest\t%s\n' "${MANIFEST}"
    printf '# manifest_sha256\t%s\n' "${MANIFEST_SHA}"
    printf '# runner_pid\t%s\n' "$$"
    printf '# queue_log\t%s\n' "${QUEUE_LOG}"
    printf '# started_utc\t%s\n' "${STARTED_UTC}"
    printf '# updated_utc\t%s\n' "$(now_utc)"
    printf '# slot\t%s of %s\n' "${CURRENT_SLOT}" "$(printf '%s\n' "${SLOTS}" | tail -n 1)"
    printf '# tally'
    for state in pending running exited-ok exited-ok-no-artifact exited-nonzero \
                 skipped-complete refused-busy-gpu; do
      printf '\t%s=%s' "${state}" "${tally[${state}]}"
    done
    printf '\n'
    printf '# FAILURES\t%s%s\n' "${n_failed}" "${failures:+: ${failures}}"
    printf '# NO-RECORD\t%s%s\n' "${n_silent}" "${silent:+: ${silent}}"
    printf 'slot\tlabel\tgpu\tstate\texit\tstarted_utc\tended_utc\tartifact\tlog\n'
    for ((i = 0; i < N; i++)); do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${C_SLOT[$i]}" "${C_LABEL[$i]}" "${C_GPU[$i]}" "${C_STATE[$i]}" "${C_EXIT[$i]}" \
        "${C_START[$i]}" "${C_END[$i]}" "${C_ART[$i]}" "${C_LOG[$i]}"
    done
  } > "${tmp}"
  mv -f "${tmp}" "${STATUS}"
}

# ------------------------------------------------------------------- helpers

# The completion test, and the artefact path when it passes. Same rule as
# run_external_baseline_h200.sh's default PRESENT_PATTERN ("\\.json\$").
cell_artifact() {
  local out_dir="$1" name
  name="$(ls -1 "${out_dir}" 2>/dev/null | grep -m 1 '\.json$' || true)"
  [ -n "${name}" ] || return 1
  printf '%s/%s\n' "${out_dir}" "${name}"
}

# Busy means the same thing it means in the driver's poll loop: more than
# BUSY_MIB of device memory in use on that card.
gpu_is_busy() {
  local index="$1" used
  used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', *' -v g="${index}" '$1==g{print $2}' | tr -dc '0-9')"
  [ -n "${used}" ] || return 1
  [ "${used}" -gt "${BUSY_MIB}" ]
}

launch_cell() {
  local i="$1"
  local snapshot="${SNAPSHOT_FOR[${C_KEY[$i]}]}"
  local out_dir="${C_OUT[$i]}" pod_log="${C_LOG[$i]}"
  mkdir -p "${out_dir}" "$(dirname "${pod_log}")"
  (
    set -euo pipefail
    export TRANSFER_PACKAGE_ROOT="${snapshot}"
    # shellcheck disable=SC1090
    source "${snapshot}/scripts/transfer/h200_env.sh"
    : "${TRANSFER_PROGEN3_DIR:?must be exported by h200_env.sh or the caller}"
    : "${TRANSFER_PROGEN3_SRC:?must be exported by h200_env.sh or the caller}"
    : "${TRANSFER_PYTHON:?must be exported by h200_env.sh or the caller}"
    if [ "${C_ENV[$i]}" != "-" ]; then
      local -a cell_env=()
      read -r -a cell_env <<< "${C_ENV[$i]}"
      export "${cell_env[@]}"
    fi
    # The one substitution this runner performs, so the model root stays
    # declared in h200_env.sh alone. The pattern is single-quoted, so it is a
    # literal and the args field is never eval'd.
    local expanded="${C_ARGS[$i]//'${TRANSFER_MODEL_BASE_DIR}'/${TRANSFER_MODEL_BASE_DIR}}"
    local -a argv=()
    read -r -a argv <<< "${expanded}"
    cd "${snapshot}"
    exec "${TRANSFER_PYTHON}" "${snapshot}/scripts/transfer/${C_STAGE[$i]}" \
      --device "cuda:${C_GPU[$i]}" --out "${out_dir}" "${argv[@]}"
  ) > "${pod_log}" 2>&1 < /dev/null &
  C_PID[$i]=$!
}

# ------------------------------------------------------------------ the queue

write_status
overall=0

for slot in ${SLOTS}; do
  CURRENT_SLOT="${slot}"
  running=()
  for ((i = 0; i < N; i++)); do
    [ "${C_SLOT[$i]}" = "${slot}" ] || continue
    if art="$(cell_artifact "${C_OUT[$i]}")"; then
      C_STATE[$i]="skipped-complete"; C_ART[$i]="${art}"
      log "slot ${slot} ${C_LABEL[$i]}: already complete at ${art}; skipping"
      continue
    fi
    if gpu_is_busy "${C_GPU[$i]}"; then
      C_STATE[$i]="refused-busy-gpu"; C_EXIT[$i]="-"
      overall=1
      log "slot ${slot} ${C_LABEL[$i]}: cuda:${C_GPU[$i]} is not idle; REFUSED (not launched)"
      continue
    fi
    C_START[$i]="$(now_utc)"; C_STATE[$i]="running"
    launch_cell "${i}"
    running+=("${i}")
    log "slot ${slot} ${C_LABEL[$i]}: launched on cuda:${C_GPU[$i]} as pid ${C_PID[$i]}"
  done
  write_status

  # The barrier. Poll rather than `wait` in manifest order, so a cell that
  # finishes early reaches the status file when it finishes and not when its
  # turn comes round.
  while [ "${#running[@]}" -gt 0 ]; do
    still=()
    for i in "${running[@]}"; do
      if kill -0 "${C_PID[$i]}" 2>/dev/null; then
        still+=("${i}")
        continue
      fi
      code=0
      wait "${C_PID[$i]}" || code=$?
      C_EXIT[$i]="${code}"; C_END[$i]="$(now_utc)"
      if [ "${code}" -ne 0 ]; then
        C_STATE[$i]="exited-nonzero"; overall=1
        log "slot ${slot} ${C_LABEL[$i]}: EXITED ${code}; see ${C_LOG[$i]}"
      elif art="$(cell_artifact "${C_OUT[$i]}")"; then
        C_STATE[$i]="exited-ok"; C_ART[$i]="${art}"
        log "slot ${slot} ${C_LABEL[$i]}: exited 0, wrote ${art}"
      else
        C_STATE[$i]="exited-ok-no-artifact"; overall=1
        log "slot ${slot} ${C_LABEL[$i]}: exited 0 but wrote no record; see ${C_LOG[$i]}"
      fi
      write_status
    done
    running=(${still[@]+"${still[@]}"})
    [ "${#running[@]}" -eq 0 ] || sleep "${POLL_SECONDS}"
  done

  log "slot ${slot} complete"
  write_status
  # Device memory is released by the kernel a moment after the process goes, so
  # the next slot's idle-card check gets a settle rather than a race.
  sleep "${SLOT_SETTLE_SECONDS}"
done

CURRENT_SLOT="done"
write_status
if [ "${overall}" -eq 0 ]; then
  log "campaign ${CAMPAIGN} complete: every cell exited 0 with a record"
else
  log "campaign ${CAMPAIGN} INCOMPLETE: see the FAILURES line of ${STATUS}"
fi
exit "${overall}"
