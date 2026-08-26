#!/usr/bin/env bash
set -euo pipefail

# ############################################################################
# VALIDATED ON THE RECORDS PATH (2026-08-17), single-file AND multi-file, the
# second only after a defect that single-file testing could not see: the per-file
# loop lost its stdin to the pull command and ran exactly one iteration. See the
# comment at that loop. Steps 1 and 2 of "First run" below passed against a
# real pod: --dry-run selected exactly the one *.json in each of two completed
# cell directories and moved nothing, and a records-only pull of a cell the
# driver had already admitted returned a byte-identical file -- 46,377 bytes,
# sha256 932e7682e132.., verified per-file by h200_gpfs_pull.sh and again by
# this script's own sha256sum -c -- in 62 s, leaving the 8.6 GB dictionary on
# GPFS. Step 3, --with-weights, has NOT been run and is still unexercised.
# ############################################################################
#
# Retrieve the RECORDS of a completed run and leave the dictionary weights on
# GPFS -- digest-verified either way.
#
# Why this exists. The scientific verdict of a dictionary cell is a ~46 KB JSON;
# the dictionary beside it is 8.6 GB at d_hidden 8192 and about 17.2 GB at
# 16384. run_external_baseline_h200.sh pulls a cell with
#
#     "${H200_CLI}" sync pull "${OUT_DIR}" "${LOCAL_OUT}"
#
# and `h200 sync pull` is a directory operation -- it tars the whole source
# directory in the pod, moves the archive, and extracts it. It is ALL-OR-NOTHING
# PER DIRECTORY: there is no way to ask it for a subset. So today the verdict
# waits behind the payload by construction, and EXP-R2-204 records that an
# 8.59 GB directory pull has failed twice on chunk-size mismatch. Under an
# unstable link that is the wrong default.
#
# The smallest change, and it does not touch the access layer. One level below
# `h200 sync`, `h200 pull` is already per-FILE and already
# digest-verified: it reads the remote size and sha256 first, short-circuits
# when the local copy already matches, stages and pulls in chunks with retries,
# and refuses on a checksum mismatch. Everything missing was file SELECTION, and
# that belongs on this side. So this script contributes no transfer code: it
# chooses the files, drives the existing per-file pull, and then applies the
# same digest comparison run_external_baseline_h200.sh admits a result on.
#
# What is a record and what is a payload. A record is a `*.json`.
# 17_train_transcoder.py writes `<stem>.pt` (the dictionary) and `<stem>.json`
# (the record); 32_crosscoder.py writes only a JSON. Nothing else is produced by
# the stages this is for, so "*.json" is a complete rule for them and is stated
# as a rule about those stages rather than as a general one.
#
# Usage:
#   export H200_POD=<running-pod-name>
#   scripts/transfer/pull_records_h200.sh <gpfs-directory> <local-directory>
#   scripts/transfer/pull_records_h200.sh --with-weights <gpfs-dir> <local-dir>
#   scripts/transfer/pull_records_h200.sh --dry-run <gpfs-dir> <local-dir>
#
# The GPFS directory may be one cell's output directory or a whole run root --
# the walk recurses and relative paths are preserved, so
#
#   pull_records_h200.sh <gpfs>/results/external_baseline/<run-id> results/transfer/external_baseline/<run-id>
#
# retrieves every cell's record in one invocation and moves a few hundred KB.
#
# Exit codes follow run_external_baseline_h200.sh's vocabulary where they
# overlap: 2 usage, 4 nothing to pull, 5 digest mismatch (NOT ADMITTED).
#
# Two limitations, recorded rather than designed around:
#   * --with-weights makes the pod sha256 a multi-gigabyte file before the first
#     byte moves, which holds the exec channel open for minutes. The weights
#     path therefore inherits the link's fragility; the records path does not,
#     because it hashes about 46 KB. That asymmetry is the point of the default.
#   * The remote listing pipes `find -printf '%P\n'` into `xargs sha256sum`,
#     which is the driver's own line and which breaks on a file name containing
#     whitespace. No stage here writes one.
#
# First run, cheapest disconfirming check first:
#   1. --dry-run against a completed cell directory: one pod round trip, no
#      transfer. It proves the pod path exists and the selection is right.
#                                                                 PASSED 08-17.
#   2. Records-only pull of the same cell, then diff the pulled JSON against the
#      copy the driver already admitted for an earlier cell of the same shape.
#                                                                 PASSED 08-17.
#   3. --with-weights on the SMALLEST completed cell, not a 17.2 GB one.
#                                                                 NOT RUN.
#   4. A MULTI-FILE selection -- a whole run root rather than one cell. Added
#      after step 2 passed on a single file and the multi-file path then failed
#      on the first real use. Any loop must be exercised with more than one
#      iteration before it is called validated.        PASSED 08-17 after the fix.

if [[ -n "${H200_ACCESS_ROOT:-}" ]]; then
  echo "H200_ACCESS_ROOT is no longer used; unset it and use HANGZHOU_COMPUTE_ROOT" >&2
  exit 2
fi
HANGZHOU_COMPUTE_ROOT="${HANGZHOU_COMPUTE_ROOT:-${HOME}/hangzhou-compute}"
H200_CLI="${H200_CLI:-${HANGZHOU_COMPUTE_ROOT}/h200}"

usage() {
  sed -n '/^# Usage:/,/^# Exit codes/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

WITH_WEIGHTS=0
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --with-weights) WITH_WEIGHTS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done

[ $# -eq 2 ] || { usage >&2; exit 2; }
REMOTE_DIR="$1"
LOCAL_DIR="$2"
: "${H200_POD:?H200_POD must be exported by the caller}"

log() { printf '[pull-records] %s %s\n' "$(date -Is)" "$*"; }

if [ "${WITH_WEIGHTS}" -eq 1 ]; then
  FIND_FILTER=""
  WHAT="every file"
else
  FIND_FILTER="-name '*.json'"
  WHAT="records (*.json)"
fi

# One pod round trip: the selected files and their digests, in the driver's own
# form (`find . -type f -printf '%P\n' | sort | xargs sha256sum`), narrowed by
# the selection. `xargs -r` so an empty selection does not run sha256sum with no
# arguments and report the whole cwd.
SUMS="$(mktemp)"
trap 'rm -f "${SUMS}"' EXIT
log "listing ${WHAT} under ${REMOTE_DIR}"
"${H200_CLI}" bash \
  "cd '${REMOTE_DIR}' && find . -type f ${FIND_FILTER} -printf '%P\n' | sort | xargs -r sha256sum" \
  > "${SUMS}"

COUNT="$(wc -l < "${SUMS}" | tr -dc '0-9')"
if [ -z "${COUNT}" ] || [ "${COUNT}" -eq 0 ]; then
  echo "no ${WHAT} under ${REMOTE_DIR}; nothing to pull" >&2
  exit 4
fi
log "${COUNT} file(s) selected"

if [ "${DRY_RUN}" -eq 1 ]; then
  awk '{print $2}' "${SUMS}"
  log "dry run; nothing transferred"
  exit 0
fi

mkdir -p "${LOCAL_DIR}"
while read -r _sha name; do
  [ -n "${name}" ] || continue
  mkdir -p "${LOCAL_DIR}/$(dirname "${name}")"
  # `< /dev/null` is load-bearing. `h200 pull` reads stdin, and inside a
  # `while read ... done < file` loop that means it consumes the loop's remaining
  # input: the first iteration ran, swallowed the other eleven lines, and the loop
  # ended silently having pulled nothing. The digest check caught it -- exit 5, NOT
  # ADMITTED, four records missing -- but only because those four were not already
  # on disk from an earlier directory pull.
  #
  # This is invisible with a one-file selection, which is exactly how it survived
  # validation: a loop exercised with a single iteration is not an exercised loop.
  "${H200_CLI}" pull "${REMOTE_DIR}/${name}" "${LOCAL_DIR}/${name}" < /dev/null
done < "${SUMS}"

# Admission is the driver's rule, unchanged: a result is admitted only if the
# digests taken on each side agree. `h200 pull` verifies each file as it
# moves; this re-checks the delivered set as a whole, so a file that never
# arrived is caught as well as one that arrived wrong.
if ( cd "${LOCAL_DIR}" && LC_ALL=C sha256sum -c "${SUMS}" >/dev/null 2>&1 ); then
  log "digests verified; ${LOCAL_DIR} ADMITTED (${COUNT} file(s))"
  if [ "${WITH_WEIGHTS}" -eq 0 ]; then
    log "dictionary weights were NOT pulled and remain on GPFS under ${REMOTE_DIR}"
  fi
else
  echo "digest mismatch between pod and B; NOT ADMITTED: ${LOCAL_DIR}" >&2
  ( cd "${LOCAL_DIR}" && LC_ALL=C sha256sum -c "${SUMS}" 2>&1 | grep -v ': OK$' >&2 ) || true
  exit 5
fi
