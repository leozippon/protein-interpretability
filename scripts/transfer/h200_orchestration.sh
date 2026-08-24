#!/usr/bin/env bash
# Shared H200 orchestration helpers. Sourced, so this file must not change the
# caller's shell options or export a pod name.
#
# One declaration for the completion, digest, host-snapshot, timeout, and
# stage-37 resource rules used by the in-pod campaign queue and the
# workstation external-baseline driver.

# 24 h is the safe default for a 12-24 h campaign. 48 h is the hard ceiling so
# TIMEOUT_SECONDS remains a bound rather than an unbounded wait.
TRANSFER_DEFAULT_TIMEOUT_SECONDS="${TRANSFER_DEFAULT_TIMEOUT_SECONDS:-86400}"
TRANSFER_MAX_TIMEOUT_SECONDS="${TRANSFER_MAX_TIMEOUT_SECONDS:-172800}"

assert_timeout_seconds() {
  local value="$1"
  case "${value}" in
    ''|*[!0-9]*)
      echo "timeout must be an integer number of seconds, got: ${value:-empty}" >&2
      return 2
      ;;
  esac
  if [ "${value}" -lt 1 ]; then
    echo "timeout must be at least 1s, got: ${value}" >&2
    return 2
  fi
  if [ "${value}" -gt "${TRANSFER_MAX_TIMEOUT_SECONDS}" ]; then
    echo "timeout ${value}s exceeds the ${TRANSFER_MAX_TIMEOUT_SECONDS}s bound" >&2
    return 2
  fi
}

# Exact JSON basename only. No directory, glob, regex, or traversal.
assert_expect_basename() {
  local name="$1"
  case "${name}" in
    *.json) ;;
    *)
      echo "expect must be a JSON basename, got: ${name:-empty}" >&2
      return 2
      ;;
  esac
  case "${name}" in
    ''|'.json'|.*|*/*|*\\*|*'~'*|*$'\n'*)
      echo "expect must be a JSON basename with no path or hidden prefix: ${name}" >&2
      return 2
      ;;
  esac
  case "${name}" in
    *'*'*|*'?'*|'['*|']'*)
      echo "expect must be an exact basename, not a glob or regex: ${name}" >&2
      return 2
      ;;
  esac
  case "${name}" in
    *[!A-Za-z0-9._-]*)
      echo "expect may contain only letters, digits, '.', '_' and '-': ${name}" >&2
      return 2
      ;;
  esac
}

expected_json_path() {
  local out_dir="$1" expect="$2"
  assert_expect_basename "${expect}" || return 2
  printf '%s/%s\n' "${out_dir}" "${expect}"
}

# Presence of the exact basename. Unrelated JSON does not count.
cell_expected_artifact() {
  local out_dir="$1" expect="$2" path
  path="$(expected_json_path "${out_dir}" "${expect}")" || return 2
  [ -f "${path}" ] || return 1
  printf '%s\n' "${path}"
}

# Nonempty valid JSON plus an atomic SHA-256 sidecar. A missing or malformed
# file is not success and must not write a digest.
admit_expected_json() {
  local path="$1" digest sidecar tmp
  [ -f "${path}" ] || return 1
  [ -s "${path}" ] || return 1
  python3 -c 'import json, sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
if not text.strip():
    raise SystemExit(1)
json.loads(text)' "${path}" || return 1
  digest="$(sha256sum "${path}" | awk '{print $1}')"
  [ -n "${digest}" ] || return 1
  sidecar="${path}.sha256"
  tmp="${sidecar}.tmp.$$"
  printf '%s  %s\n' "${digest}" "$(basename "${path}")" > "${tmp}"
  mv -f "${tmp}" "${sidecar}"
  printf '%s\n' "${digest}"
}

# Durable host snapshot. Query form only: no hostname, no pod name.
write_host_resource_snapshot() {
  local dest="$1" label="${2:-}" tmp
  mkdir -p "$(dirname "${dest}")"
  tmp="${dest}.tmp.$$"
  {
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ -n "${label}" ]; then
      printf 'label=%s\n' "${label}"
    fi
    printf '%s\n' '--- nvidia-smi ---'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv \
      || printf '%s\n' 'nvidia-smi FAILED'
    printf '%s\n' '--- free -h ---'
    free -h || printf '%s\n' 'free FAILED'
  } > "${tmp}"
  mv -f "${tmp}" "${dest}"
}

# Stage 37's UniRef50 backgrounds are a required-variable contract, not a
# hard-coded host path. Other stages have nothing extra to declare here.
require_stage_resources() {
  local stage="$1"
  case "${stage}" in
    37_alphabet_chemistry.py)
      if [ -z "${TRANSFER_KMER_BACKGROUND_DIR:-}" ]; then
        echo "stage 37 requires TRANSFER_KMER_BACKGROUND_DIR" >&2
        return 2
      fi
      if [ -z "${TRANSFER_HIGH_ORDER_BACKGROUND_DIR:-}" ]; then
        echo "stage 37 requires TRANSFER_HIGH_ORDER_BACKGROUND_DIR" >&2
        return 2
      fi
      if [ ! -e "${TRANSFER_KMER_BACKGROUND_DIR}" ]; then
        echo "missing TRANSFER_KMER_BACKGROUND_DIR: ${TRANSFER_KMER_BACKGROUND_DIR}" >&2
        return 2
      fi
      if [ ! -e "${TRANSFER_HIGH_ORDER_BACKGROUND_DIR}" ]; then
        echo "missing TRANSFER_HIGH_ORDER_BACKGROUND_DIR: ${TRANSFER_HIGH_ORDER_BACKGROUND_DIR}" >&2
        return 2
      fi
      ;;
  esac
}

# In-pod wrapper for one external-baseline stage. Must not exec: a successful
# exec would replace this shell and drop the EXIT trap, so the post snapshot
# would never be written. Cleanup is installed first so a preflight failure
# still leaves a terminal post record.
run_wrapped_external_stage() {
  # Trap-visible state cannot be `local`: EXIT/TERM/INT run after `exit` and
  # would otherwise see empty locals.
  _XFER_WRAP_STAGE_STATUS=0
  _XFER_WRAP_POST_STATUS=0
  _XFER_WRAP_POST_WRITTEN=0
  _XFER_WRAP_CHILD=""
  _XFER_WRAP_INTERRUPTED=0

  _wrapper_write_post() {
    if [ "${_XFER_WRAP_POST_WRITTEN}" -eq 1 ]; then
      return 0
    fi
    _XFER_WRAP_POST_WRITTEN=1
    if [ -n "${_XFER_WRAP_CHILD}" ]; then
      kill "${_XFER_WRAP_CHILD}" 2>/dev/null || true
      wait "${_XFER_WRAP_CHILD}" 2>/dev/null || true
      _XFER_WRAP_CHILD=""
    fi
    write_host_resource_snapshot "${XFER_HOST_POST}" post || _XFER_WRAP_POST_STATUS=$?
    if [ "${_XFER_WRAP_POST_STATUS}" -ne 0 ] && [ "${_XFER_WRAP_STAGE_STATUS}" -eq 0 ] && [ "${_XFER_WRAP_INTERRUPTED}" -eq 0 ]; then
      _XFER_WRAP_STAGE_STATUS=2
    fi
  }

  trap _wrapper_write_post EXIT
  trap '_XFER_WRAP_INTERRUPTED=143; _XFER_WRAP_STAGE_STATUS=143; _wrapper_write_post; exit 143' TERM
  trap '_XFER_WRAP_INTERRUPTED=130; _XFER_WRAP_STAGE_STATUS=130; _wrapper_write_post; exit 130' INT

  : "${XFER_STAGE:?XFER_STAGE must name the stage file}"
  : "${XFER_OUT:?XFER_OUT must name the output directory}"
  : "${XFER_GPU:?XFER_GPU must name the card index}"
  : "${XFER_HOST_PRE:?XFER_HOST_PRE must name the pre snapshot}"
  : "${XFER_HOST_POST:?XFER_HOST_POST must name the post snapshot}"
  : "${TRANSFER_PYTHON:?TRANSFER_PYTHON must be exported}"

  require_stage_resources "$(basename "${XFER_STAGE}")" || {
    _XFER_WRAP_STAGE_STATUS=$?
    exit "${_XFER_WRAP_STAGE_STATUS}"
  }
  write_host_resource_snapshot "${XFER_HOST_PRE}" pre || {
    _XFER_WRAP_STAGE_STATUS=$?
    exit "${_XFER_WRAP_STAGE_STATUS}"
  }

  set +e
  "${TRANSFER_PYTHON}" "${XFER_STAGE}" --device "cuda:${XFER_GPU}" --out "${XFER_OUT}" "$@" &
  _XFER_WRAP_CHILD=$!
  wait "${_XFER_WRAP_CHILD}"
  _XFER_WRAP_STAGE_STATUS=$?
  _XFER_WRAP_CHILD=""
  set -e
  _wrapper_write_post
  exit "${_XFER_WRAP_STAGE_STATUS}"
}
