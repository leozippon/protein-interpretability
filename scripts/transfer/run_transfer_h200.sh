#!/usr/bin/env bash
set -euo pipefail

# Controller for the H200 transfer-measurement campaign. Runs on the
# local L20 host, where the git checkout lives. It does not run any
# experiment itself -- it freezes a code snapshot, ships it to GPFS, and
# hands off to the WORKER (h200_worker.sh), which runs inside an H200 pod
# against GPFS-mounted models and data. See scripts/transfer/README.md for
# the full controller/worker explanation and docs/methods/
# TRANSFER_MEASUREMENT_PROGRAMME.md for what each stage measures.
#
# Compute policy: L20 (local, 8x46GB) is validation only -- invoke one
# scripts/transfer/0X_*.py directly with a small cohort, never through this
# controller. H200 (remote, 16x80GB total; 4 idle H200s, 143 GB each, on the
# pod this was last checked against) is for full-scale campaigns and is what
# this controller targets. GPU count is not assumed: the worker derives it
# from nvidia-smi at run time, because pods are disposable and the next one
# may not match.
#
# Why a code freeze. A run must be bound to an immutable snapshot of
# everything that determines its result, so that result can never be
# attributed to code that has since changed underneath it. The frozen set
# is src/transfer/ and scripts/transfer/ in full, plus the transitive
# closure of every `src.*` module those files import (see freeze_manifest)
# -- not a hard-coded directory list, after a directory-list freeze once
# shipped a snapshot missing src/revision/ entirely and failed a real
# campaign at tier 1. This script computes one content hash over that
# whole closure (excluding __pycache__), derives a run-id from that hash
# and a timestamp, pushes a copy of the closure to a versioned GPFS path
# named after that run-id, and refuses to reuse a run-id whose GPFS
# directory already holds different content. The worker that actually
# executes is the one that shipped inside that same snapshot, not a
# separately-deployed copy, so a later fix to the worker cannot
# retroactively change what an old run-id ran.
#
# H200 access (privacy: no infrastructure identifiers belong in this
# file). Pods are disposable; this controller never defaults to one. Set
# H200_POD to a running pod before invoking this script (see
# ~/hangzhou-remote/README.md). Before doing anything else, this script
# invokes the documented cluster health check
# (~/hangzhou-remote/ssh_tunnel/h200_status.sh) and aborts if it fails,
# rather than reimplementing tunnel/master/node/pod checks itself. All
# GPFS access goes through the documented access-layer tools
# (h200_sync.sh, h200_gpfs_push.sh, h200_pod_bash.sh, h200_pod_exec.sh) --
# this script never opens its own connection to the cluster.
#
# Usage:
#   export H200_POD=<running-pod-name>
#   bash scripts/transfer/run_transfer_h200.sh --dry-run
#   bash scripts/transfer/run_transfer_h200.sh
#   ARMS=gpt2-large,protgpt2 GPUS=0,1 bash scripts/transfer/run_transfer_h200.sh
#   RUN_ID=20260728120000_ab12cd34ef56 bash scripts/transfer/run_transfer_h200.sh
#     (reuses an existing frozen snapshot instead of freezing a new one;
#     only valid if that run-id's embedded hash matches the code on disk
#     right now -- see resolve_run_id below)
#
# All paths and lists are overridable via environment variables; run with
# --help (or read the config block below) for the full list.

H200_ACCESS_ROOT="${H200_ACCESS_ROOT:-${HOME}/hangzhou-remote}"
H200_STATUS_CHECK="${H200_STATUS_CHECK:-${H200_ACCESS_ROOT}/ssh_tunnel/h200_status.sh}"
H200_SYNC="${H200_SYNC:-${H200_ACCESS_ROOT}/ssh_tunnel/h200_sync.sh}"
H200_GPFS_PUSH="${H200_GPFS_PUSH:-${H200_ACCESS_ROOT}/ssh_tunnel/h200_gpfs_push.sh}"
H200_POD_BASH="${H200_POD_BASH:-${H200_ACCESS_ROOT}/ssh_tunnel/h200_pod_bash.sh}"
H200_POD_EXEC="${H200_POD_EXEC:-${H200_ACCESS_ROOT}/ssh_tunnel/h200_pod_exec.sh}"

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${CONTROLLER_DIR}/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"

# The campaign panel, the stage list and each stage's eligible arms, generated
# from src/transfer/arms.py by scripts/transfer/panel_contract.py --emit. Sourced
# rather than restated: this file and h200_worker.sh each used to carry their own
# copy of the arm list, and the worker carried three further hand-written arm
# groupings besides. This script deliberately does NOT import the Python
# declaration directly -- the freeze step is required to work on a host without
# torch importable -- so the worker re-verifies the generated file against the
# live panel inside the pod, before any GPU is scheduled.
CONTRACT_SH="${CONTROLLER_DIR}/panel_contract.sh"
if [ ! -f "${CONTRACT_SH}" ]; then
  echo "missing ${CONTRACT_SH}; run: python scripts/transfer/panel_contract.py --emit" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${CONTRACT_SH}"

# GPFS is only reachable through the access-layer tools above; every path
# below is a remote path passed to them, never touched directly by this
# script.
GPFS_PROJECT_ROOT="${GPFS_PROJECT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer}"
GPFS_PACKAGE_ROOT="${GPFS_PACKAGE_ROOT:-${GPFS_PROJECT_ROOT}/packages}"
GPFS_RESULTS_ROOT="${GPFS_RESULTS_ROOT:-${GPFS_PROJECT_ROOT}/results}"
GPFS_LOGS_ROOT="${GPFS_LOGS_ROOT:-${GPFS_PROJECT_ROOT}/logs}"

# The campaign panel, not a hand-written subset of it. This line used to read
# ARMS="${ARMS:-gpt2-large,protgpt2,zymctrl,progen2-medium}" -- four of the
# eleven arms TRANSFER_CAMPAIGN_PANEL declares -- so a default campaign measured a
# four-arm panel through every stage while each stage's arm list was, correctly,
# intersected with it. That is the L18 failure verbatim, and it is the failure
# panel_contract.py was built to end: the contract removed the duplicated arm
# lists but left the default that narrowed them. Narrowing is still available,
# it just has to be asked for: ARMS=gpt2-large,protgpt2 bash run_transfer_h200.sh
ARMS="${ARMS:-${TRANSFER_CAMPAIGN_PANEL// /,}}"
GPUS="${GPUS:-0,1,2,3}"
TEXT_ARM="${TEXT_ARM:-gpt2-large}"
# Empty means "derive from nvidia-smi inside the pod, no fixed assertion" --
# see h200_worker.sh's verify_gpus. Pods are disposable; do not hard-code a
# count here. Set explicitly only to add an extra minimum-count assertion.
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-}"
# Soft warning only in the worker, not a gate: unprofiled placeholder.
# Observed L20 validation peaks were 1.6-12.2 GiB per arm.
MIN_FREE_MEM_MIB="${MIN_FREE_MEM_MIB:-16000}"
FORCE="${FORCE:-0}"

# Per-stage scale-parameter passthrough. This is the ONLY way to reach a
# stage script's own --n-seq/--pool-size/--seeds/etc: the worker otherwise
# runs every script at its own built-in defaults, which are validation-scale
# (see scripts/transfer/README.md's "Environment contract"). Each script
# names its scale knobs differently, so this is one raw-flag-string variable
# per stage rather than an enumerated set of options; e.g.
#   ARGS_PATHWAY_BUDGET="--n-seq 500 --pool-size 1000"
# Empty (the default) changes nothing -- every stage runs at whatever
# scripts/transfer/0X_*.py itself defaults to.
#
# ARGS_<STAGE>__<ITEM> is the same thing scoped to one item of one stage, with
# the item name upper-cased and every non-alphanumeric character replaced by an
# underscore -- e.g. ARGS_COHORT_POWER__PROTEIN_PROGEN2_MEDIUM="--n-seq 100", or
# ARGS_LENS_FAMILY__PROGEN2_MEDIUM="--n-seq 64". It exists because ARGS_<STAGE>
# reaches every item of a stage at once, and cohort_power's four items differ in
# exactly the ways that make one scale knob wrong for the others. The worker
# refuses either kind if it repeats a flag the worker already sets for that item,
# rather than letting argparse take the last one silently.
#
# The panel contract supplies the stage list, so a new stage is picked up here
# without this file being edited.
IFS=' ' read -r -a STAGE_NAMES <<< "${TRANSFER_STAGE_ORDER}"

# Which stages actually run this campaign, comma-separated. Default is every
# stage the contract declares; staging is often partial (some data or model rungs
# not landed yet), so a campaign scoped to what is staged today is a normal way
# to run this, not a workaround. Validated against STAGE_NAMES below.
STAGES="${STAGES:-$(IFS=,; echo "${STAGE_NAMES[*]}")}"

RUN_ID="${RUN_ID:-}"
DRY_RUN=0

CONTROLLER_LOG_DIR="${PROJECT_ROOT}/logs/transfer_h200_controller"

# ------------------------------------------------------------------ helpers

usage() {
  cat <<'EOF'
Usage: H200_POD=<pod> run_transfer_h200.sh [--dry-run] [--force]

Environment overrides: H200_POD (required except for --help), H200_ACCESS_ROOT,
REPO_ROOT, PROJECT_ROOT, GPFS_PROJECT_ROOT, GPFS_PACKAGE_ROOT,
GPFS_RESULTS_ROOT, GPFS_LOGS_ROOT, ARMS (comma-separated), GPUS
(comma-separated indices, pod-relative), TEXT_ARM, STAGES (comma-separated,
default every stage the panel contract declares -- scope a campaign to what
is staged today), ARGS_<STAGE> (raw extra CLI args per stage, e.g.
ARGS_PATHWAY_BUDGET="--n-seq 500"), ARGS_<STAGE>__<ITEM> (the same for one
item, e.g. ARGS_COHORT_POWER__PROTEIN_PROGEN2_MEDIUM="--n-seq 100"),
EXPECTED_GPU_COUNT (optional extra minimum-count assertion; empty means the
worker trusts nvidia-smi alone), MIN_FREE_MEM_MIB (soft warning only in the
worker), FORCE (0/1), RUN_ID (reuse an existing frozen snapshot instead of
freezing a new one). See scripts/transfer/README.md's "Environment
contract" for the full table.

The pod name is never written to stdout, to the controller log or to the run
manifest; see `redact`.
EOF
}

# Pod-name redaction, applied to everything this script emits.
#
# Standing rule: never persist pod names in repository files or durable run
# records. It used to be
# printed outright by the startup banner and could also reach a log through any
# path or message that happened to embed it -- a subprocess error, an access-layer
# tool's own diagnostic, a GPFS path an operator had named after a pod. Relying on
# the operator to pipe the run through `sed` makes the guarantee depend on how the
# script was invoked, which is not a guarantee.
#
# So redaction is structural: `redact` is applied to this script's own log lines
# and, in invoke_worker, to the worker's entire merged stdout/stderr before it
# reaches either the terminal or the controller log. H200_POD is exported for the
# access-layer tools (which need it) but is never itself printed.
redact() {
  if [ -z "${H200_POD:-}" ]; then
    cat
  else
    sed -e "s|${H200_POD}|<pod-redacted>|g"
  fi
}

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | redact
}

require_local_path() {
  local label="$1" path="$2"
  if [ ! -e "${path}" ]; then
    echo "missing required ${label}: ${path}" >&2
    exit 2
  fi
}

# Populates two globals from the ARGS_<STAGE> environment variables: a
# human-readable summary line (for logs, --dry-run and the run manifest)
# and STAGE_ARGS_FLAGS, the actual `--stage-args STAGE BASE64` pairs to
# hand the worker. Base64-encoded because the value crosses the
# controller -> h200_pod_exec.sh -> kubectl exec -> worker argv boundary,
# and a plain string would need to survive re-splitting at each hop; base64
# makes each one a single opaque token regardless of what it contains.
collect_stage_args() {
  STAGE_ARGS_FLAGS=()
  STAGE_ARGS_SUMMARY=""
  # Newline-separated "scope<TAB>value" records, so the run manifest records
  # exactly what was collected. It used to re-read nine named ARGS_* variables
  # from the environment, which meant the two stages that had no named variable
  # (homology_control, induction_path_patching) could be passed to the worker and
  # not recorded, and a stage added to STAGE_NAMES would silently join them.
  STAGE_ARGS_RECORDS=""
  local stage item var value b64 item_var supplied
  declare -A allowed_vars=()
  for stage in "${STAGE_NAMES[@]}"; do
    var="ARGS_$(printf '%s' "${stage}" | tr '[:lower:]' '[:upper:]')"
    allowed_vars["${var}"]=1
    for item in $(items_for_stage "${stage}"); do
      item_var="${var}__$(printf '%s' "${item}" | tr '[:lower:]' '[:upper:]' | tr -c '[:alnum:]' '_')"
      allowed_vars["${item_var%_}"]=1
    done
  done
  while IFS= read -r supplied; do
    case "${supplied}" in
      ARGS_*)
        if [ -z "${allowed_vars[${supplied}]+x}" ]; then
          echo "unknown stage-argument environment variable: ${supplied}" >&2
          exit 2
        fi
        ;;
    esac
  done < <(compgen -e)

  for stage in "${STAGE_NAMES[@]}"; do
    var="ARGS_$(printf '%s' "${stage}" | tr '[:lower:]' '[:upper:]')"
    value="${!var:-}"
    if [ -n "${value}" ]; then
      b64="$(printf '%s' "${value}" | base64 -w0)"
      STAGE_ARGS_FLAGS+=(--stage-args "${stage}" "${b64}")
      STAGE_ARGS_SUMMARY="${STAGE_ARGS_SUMMARY}${stage}=[${value}] "
      STAGE_ARGS_RECORDS="${STAGE_ARGS_RECORDS}${stage}"$'\t'"${value}"$'\n'
    fi
    # Item-scoped overrides. The item name space is the stage's own: arm names
    # for a per-arm stage, cohort_power's four item labels, and the literal
    # "panel" for a panel-wide stage. Both are enumerated from the contract so
    # that ARGS_LENS_FAMILY__QWEN2_5_0_5B -- an item that stage cannot run -- is
    # simply never collected rather than passed through to be ignored.
    for item in $(items_for_stage "${stage}"); do
      item_var="${var}__$(printf '%s' "${item}" | tr '[:lower:]' '[:upper:]' | tr -c '[:alnum:]' '_')"
      item_var="${item_var%_}"
      value="${!item_var:-}"
      if [ -n "${value}" ]; then
        b64="$(printf '%s' "${value}" | base64 -w0)"
        STAGE_ARGS_FLAGS+=(--item-args "${stage}" "${item}" "${b64}")
        STAGE_ARGS_SUMMARY="${STAGE_ARGS_SUMMARY}${stage}/${item}=[${value}] "
        STAGE_ARGS_RECORDS="${STAGE_ARGS_RECORDS}${stage}/${item}"$'\t'"${value}"$'\n'
      fi
    done
  done
}

# The item labels one stage dispatches over, from the panel contract.
items_for_stage() {
  local stage="$1"
  case "${TRANSFER_STAGE_SCOPE[${stage}]:-}" in
    armless) : ;;
    panel_wide)
      if [ "${stage}" = cohort_power ]; then
        printf '%s\n' ${TRANSFER_COHORT_ITEMS}
      else
        printf 'panel\n'
      fi
      ;;
    per_arm|control_anchored) printf '%s\n' ${TRANSFER_STAGE_ARMS[${stage}]} ;;
  esac
}

reject_duplicate_values() {
  local label="$1"
  shift
  local value
  declare -A seen=()
  for value in "$@"; do
    if [ -n "${seen[${value}]+x}" ]; then
      echo "${label} contains duplicate value: ${value}" >&2
      exit 2
    fi
    seen["${value}"]=1
  done
}

# --------------------------------------------------------------- preflight

verify_h200_cluster() {
  if [ ! -x "${H200_STATUS_CHECK}" ]; then
    echo "H200 cluster health check not found or not executable: ${H200_STATUS_CHECK}" >&2
    echo "see ~/hangzhou-remote/README.md; this controller will not schedule without it" >&2
    exit 2
  fi
  log "running H200 cluster health check (tunnel, master, nodes, GPU allocation, pod match)"
  if ! "${H200_STATUS_CHECK}"; then
    echo "H200 cluster health check failed; aborting before touching anything" >&2
    exit 2
  fi
  log "H200 cluster health check passed"
}

# ------------------------------------------------------------ code freeze

# Populates FILE_LIST with every repo-relative runtime file to stage. The
# checksum manifest is intentionally created from the staged copy afterwards.
#
# The frozen set is src/transfer/ and scripts/transfer/ (the baseline,
# unconditional) PLUS the transitive closure of every `src.*` import found
# by statically parsing that baseline -- not a second hard-coded directory
# list. Run 20260728150714_b613d3afe620 failed at tier 1, two seconds in,
# on every cohort_power item: `ModuleNotFoundError: No module named
# 'src.revision'`. The transfer package then imported src.revision.io,
# .statistics, .dictionary_fidelity and .nested_recoverability, none of
# which is under either baseline directory, so a directory-list freeze
# silently shipped an incomplete, non-reproducible snapshot. Two earlier
# campaigns did not catch it because they ran only circuit_primitives,
# which happened not to import src.revision -- an accident of stage
# selection, not evidence the freeze was complete.
#
# As of EXP-R2-066 the closure is exactly src/transfer/ plus src/__init__.py:
# the twelve symbols the package used from src/revision/ were vendored into
# src/transfer/{io,statistics,scoring}.py and the retired package was
# archived. That does NOT make this derivation redundant -- it makes it the
# only thing standing between a future dependency and the same failure, and
# it is what proved the closure was clean rather than assumed. Keep it.
#
# A syntax error anywhere in the baseline (for instance in a file another
# agent is mid-edit on) fails this step loudly via the python3 subprocess's
# own non-zero exit under `set -e` -- intentionally not caught or worked
# around, since a file that cannot even be parsed cannot be frozen.
freeze_manifest() {
  local scratch="$1"
  FILE_LIST="${scratch}/file_list.txt"
  local baseline_list="${scratch}/baseline_list.txt"
  local closure_extra="${scratch}/closure_extra.txt"

  ( cd "${PROJECT_ROOT}" && \
    find src/transfer scripts/transfer \
      \( -path '*/__pycache__/*' -o -name '*.pyc' -o -name '*.pyo' \) -prune \
      -o -type f -print
  ) | LC_ALL=C sort > "${baseline_list}"

  if [ ! -s "${baseline_list}" ]; then
    echo "code freeze found no files under src/transfer or scripts/transfer" >&2
    exit 2
  fi

  ( cd "${PROJECT_ROOT}" && python3 - "${baseline_list}" ) > "${closure_extra}" <<'PY'
import ast
import sys
from pathlib import Path

package_root = Path(".").resolve()
baseline = [
    Path(line.strip())
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
baseline_py = [p for p in baseline if p.suffix == ".py"]


def enclosing_package(path: Path) -> str:
    """The dotted package a file's relative imports (`from .. import X`) are
    resolved against -- the dotted path of its own containing directory,
    which is correct for both a regular module and that package's own
    __init__.py (Python's __package__ is the same for either)."""
    parts = path.resolve().parent.relative_to(package_root).parts
    return ".".join(parts)


def local_imports(path: Path) -> set[str]:
    """Every `src`-rooted module named in an import statement in path,
    absolute (`from src.transfer.io import write_json`, as scripts/transfer
    uses) or relative (`from .statistics import mean_interval`, as
    src/transfer/*.py uses). Both forms are required: an absolute-only scan
    silently missed two relatively-imported modules the first time this was
    written, and a relative-only one would miss every entry point.

    Static, not executed: this must work without torch/transformers
    importable (the controller runs on the local L20 host, not the pod),
    and a syntax error here is a real defect in the frozen tree, so it is
    allowed to raise and abort the freeze rather than being caught.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and (node.module == "src" or node.module.startswith("src.")):
                    found.add(node.module)
                continue
            # Relative import: resolve against this file's own package.
            # https://docs.python.org/3/reference/import.html#package-relative-imports
            pkg = enclosing_package(path)
            bits = pkg.rsplit(".", node.level - 1)
            base = bits[0]
            if base != "src" and not base.startswith("src."):
                continue
            if node.module:
                found.add(f"{base}.{node.module}")
            else:
                # `from .. import name1, name2` -- each name may be a
                # submodule of base; module_file() below only keeps it if a
                # matching file actually exists, so a name that is really
                # just an attribute of base/__init__.py is harmlessly
                # discarded rather than mis-frozen.
                for alias in node.names:
                    found.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src" or alias.name.startswith("src."):
                    found.add(alias.name)
    return found


def module_file(module: str) -> Path | None:
    candidate = package_root.joinpath(*module.split("."))
    if candidate.is_dir():
        init = candidate / "__init__.py"
        return init if init.is_file() else None
    py = candidate.with_suffix(".py")
    return py if py.is_file() else None


def init_chain(module: str) -> list[Path]:
    """Every package __init__.py on the way to `module` (e.g. src/__init__.py
    and src/transfer/__init__.py for module "src.transfer.io"), since a
    package cannot import without them."""
    parts = module.split(".")
    chain = []
    for depth in range(1, len(parts) + 1):
        init = package_root.joinpath(*parts[:depth], "__init__.py")
        if init.is_file():
            chain.append(init)
    return chain


baseline_set = set(baseline)
discovered: set[Path] = set(baseline_py)
queue: list[Path] = list(baseline_py)
seen_modules: set[str] = set()
while queue:
    current = queue.pop()
    for module in local_imports(current):
        if module in seen_modules:
            continue
        seen_modules.add(module)
        candidates = [*init_chain(module)]
        resolved = module_file(module)
        if resolved is not None:
            candidates.append(resolved)
        for candidate in candidates:
            rel = candidate.relative_to(package_root)
            if rel not in discovered:
                discovered.add(rel)
                queue.append(candidate)

extra = sorted(str(p) for p in discovered if p not in baseline_set)
for path in extra:
    print(path)
PY

  {
    cat "${baseline_list}" "${closure_extra}"
    printf '%s\n' docs/analysis/MODEL_LADDER_20260728.md
  } | LC_ALL=C sort -u > "${FILE_LIST}"
  if [ -s "${closure_extra}" ]; then
    log "import closure added $(wc -l < "${closure_extra}") file(s) outside src/transfer and scripts/transfer:"
    while IFS= read -r extra_path; do
      log "  + ${extra_path}"
    done < "${closure_extra}"
  fi

}

# Copies exactly FILE_LIST into a clean local staging directory, then creates
# and verifies the checksum manifest from those staged bytes.
stage_snapshot() {
  local scratch="$1"
  STAGING_DIR="${scratch}/staging"
  mkdir -p "${STAGING_DIR}"
  local rel
  while IFS= read -r rel; do
    mkdir -p "${STAGING_DIR}/$(dirname "${rel}")"
    cp -p "${PROJECT_ROOT}/${rel}" "${STAGING_DIR}/${rel}"
  done < "${FILE_LIST}"
  MANIFEST="${STAGING_DIR}/CODE_CONTENT_SHA256SUMS"
  ( cd "${STAGING_DIR}" && xargs -a "${FILE_LIST}" -d '\n' sha256sum ) > "${MANIFEST}"
  CODE_HASH="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
  verify_local_snapshot
}

verify_local_snapshot() {
  if ! (
    cd "${STAGING_DIR}"
    printf '%s  %s\n' "${CODE_HASH}" CODE_CONTENT_SHA256SUMS | sha256sum -c - >/dev/null
    sha256sum -c -- CODE_CONTENT_SHA256SUMS >/dev/null
  ); then
    echo "local staged snapshot failed checksum verification" >&2
    exit 2
  fi
}

# Resolves RUN_ID and SNAPSHOT_DIR. If the operator supplied RUN_ID, its
# trailing hash segment must match CODE_HASH computed just now -- resuming a
# run-id under different code is refused, not silently allowed.
resolve_run_id() {
  local short_hash="${CODE_HASH:0:12}"
  if [ -n "${RUN_ID}" ]; then
    case "${RUN_ID}" in
      *_"${short_hash}") ;;
      *)
        echo "RUN_ID=${RUN_ID} does not end in the current code hash (${short_hash});" >&2
        echo "the code on disk has changed since that run-id was minted, so it cannot be resumed" >&2
        exit 2
        ;;
    esac
  else
    RUN_ID="$(date +%Y%m%d%H%M%S)_${short_hash}"
  fi
  SNAPSHOT_DIR="${GPFS_PACKAGE_ROOT}/${RUN_ID}"
}

# Reuse requires the existing GPFS tree to match the locally staged manifest.
# Accepted unresolved defect: snapshot check/push has no GPFS lease, so two
# controllers targeting one run-id can race. A correct lock needs a broader
# owner/expiry contract; this change deliberately does not add a partial lock.
check_snapshot_absence() {
  local reply
  reply="$("${H200_POD_BASH}" "test -e '${SNAPSHOT_DIR}' && echo EXISTS || echo ABSENT")"
  case "${reply}" in
    ABSENT) SNAPSHOT_EXISTS=0 ;;
    EXISTS)
      SNAPSHOT_EXISTS=1
      verify_remote_snapshot
      ;;
    *)
      echo "could not determine whether ${SNAPSHOT_DIR} exists on GPFS (got: ${reply})" >&2
      exit 2
      ;;
  esac
}

verify_remote_snapshot() {
  local snapshot_q
  printf -v snapshot_q '%q' "${SNAPSHOT_DIR}"
  if ! "${H200_POD_BASH}" \
      "cd -- ${snapshot_q} && printf '%s  %s\n' '${CODE_HASH}' CODE_CONTENT_SHA256SUMS | sha256sum -c - >/dev/null && sha256sum -c -- CODE_CONTENT_SHA256SUMS >/dev/null"; then
    echo "snapshot checksum verification failed on GPFS: ${SNAPSHOT_DIR}" >&2
    exit 2
  fi
}

push_snapshot() {
  log "pushing code snapshot: ${STAGING_DIR} -> ${SNAPSHOT_DIR}"
  "${H200_SYNC}" push "${STAGING_DIR}" "${SNAPSHOT_DIR}"
  verify_remote_snapshot
  log "snapshot pushed: run_id=${RUN_ID} code_hash=${CODE_HASH}"
}

# ------------------------------------------------------------- run manifest

write_run_manifest() {
  local scratch="$1"
  local git_revision git_dirty
  RUN_MANIFEST="${scratch}/RUN_MANIFEST.json"

  git_revision="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [ "${git_revision}" = unknown ]; then
    git_dirty=unknown
  elif [ -n "$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null)" ]; then
    git_dirty=true
  else
    git_dirty=false
  fi

  RUN_ID="${RUN_ID}" CODE_HASH="${CODE_HASH}" MANIFEST_PATH="${MANIFEST}" \
  GIT_REVISION="${git_revision}" GIT_DIRTY="${git_dirty}" \
  ARMS="${ARMS}" GPUS="${GPUS}" TEXT_ARM="${TEXT_ARM}" STAGES="${STAGES}" \
  EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT}" MIN_FREE_MEM_MIB="${MIN_FREE_MEM_MIB}" \
  FORCE="${FORCE}" GPFS_RESULTS_ROOT="${GPFS_RESULTS_ROOT}" \
  GPFS_LOGS_ROOT="${GPFS_LOGS_ROOT}/${RUN_ID}" SNAPSHOT_DIR="${SNAPSHOT_DIR}" \
  RUN_MANIFEST_OUT="${RUN_MANIFEST}" \
  STAGE_ARGS_RECORDS="${STAGE_ARGS_RECORDS}" \
  PANEL_CONTRACT_SH="${CONTRACT_SH}" \
  python3 - "${STAGE_ORDER_CSV}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

stage_order = sys.argv[1].split(",")
per_file = []
manifest_path = Path(os.environ["MANIFEST_PATH"])
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    digest, path = line.split(None, 1)
    per_file.append({"sha256": digest, "path": path})

payload = {
    "schema_version": "interpretability_transfer_h200_invocation_v1",
    "run_id": os.environ["RUN_ID"],
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "code_hash": os.environ["CODE_HASH"],
    "code_hash_algorithm": "sha256_of_sorted_sha256sum_manifest",
    "code_content_files": per_file,
    "frozen_scope": {
        "baseline_directories": ["src/transfer", "scripts/transfer"],
        "closure_derived": True,
        "description": (
            "baseline_directories in full, plus the transitive closure of "
            "every src.* import found by static analysis of that baseline "
            "(see freeze_manifest in this script); code_content_files is "
            "the exact resulting file set actually hashed and pushed"
        ),
    },
    "research_root_git_revision": os.environ["GIT_REVISION"],
    "research_root_git_dirty": os.environ["GIT_DIRTY"],
    "snapshot_dir": os.environ["SNAPSHOT_DIR"],
    "results_root": os.environ["GPFS_RESULTS_ROOT"],
    "logs_root": os.environ["GPFS_LOGS_ROOT"],
    "stage_order": stage_order,
    "parameters": {
        "arms": os.environ["ARMS"].split(","),
        "gpus": os.environ["GPUS"].split(","),
        "text_arm": os.environ["TEXT_ARM"],
        "stages": os.environ["STAGES"].split(","),
        "expected_gpu_count": (
            int(os.environ["EXPECTED_GPU_COUNT"])
            if os.environ["EXPECTED_GPU_COUNT"]
            else None
        ),
        "min_free_mem_mib": int(os.environ["MIN_FREE_MEM_MIB"]),
        "force": os.environ["FORCE"] == "1",
        # Exactly the scale-parameter overrides collect_stage_args handed the
        # worker, keyed by "stage" or "stage/item". Recorded from what was
        # collected rather than re-read from a fixed list of environment
        # variables, so an override the worker received can never be missing here.
        "stage_args": dict(
            line.split("\t", 1)
            for line in os.environ["STAGE_ARGS_RECORDS"].splitlines()
            if line
        ),
    },
    "panel_contract_sha256": hashlib.sha256(
        Path(os.environ["PANEL_CONTRACT_SH"]).read_bytes()
    ).hexdigest(),
}
Path(os.environ["RUN_MANIFEST_OUT"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

push_run_manifest() {
  local digest destination snapshot_q destination_q
  digest="$(sha256sum "${RUN_MANIFEST}" | awk '{print $1}')"
  destination="${SNAPSHOT_DIR}/INVOCATIONS/${digest}.json"
  printf -v snapshot_q '%q' "${SNAPSHOT_DIR}"
  printf -v destination_q '%q' "${destination}"
  "${H200_POD_BASH}" "mkdir -p -- ${snapshot_q}/INVOCATIONS"
  if "${H200_POD_BASH}" "test -e ${destination_q}"; then
    if ! "${H200_POD_BASH}" "printf '%s  %s\n' '${digest}' ${destination_q} | sha256sum -c - >/dev/null"; then
      echo "existing invocation manifest failed checksum verification: ${destination}" >&2
      exit 2
    fi
    log "invocation manifest already present and verified: ${destination}"
    return
  fi
  log "pushing append-only invocation manifest -> ${destination}"
  "${H200_GPFS_PUSH}" "${RUN_MANIFEST}" "${destination}"
  if ! "${H200_POD_BASH}" "printf '%s  %s\n' '${digest}' ${destination_q} | sha256sum -c - >/dev/null"; then
    echo "invocation manifest failed checksum verification after transfer: ${destination}" >&2
    exit 2
  fi
}

# ------------------------------------------------------------- worker call

invoke_worker() {
  local force_flag=()
  [ "${FORCE}" = "1" ] && force_flag=(--force)
  mkdir -p "${CONTROLLER_LOG_DIR}"
  local controller_log="${CONTROLLER_LOG_DIR}/${RUN_ID}.log"
  log "invoking worker inside pod, run_id=${RUN_ID}; controller-side copy: ${controller_log}"
  "${H200_POD_EXEC}" -- \
    bash "${SNAPSHOT_DIR}/scripts/transfer/h200_worker.sh" \
    --run-id "${RUN_ID}" \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    --results-root "${GPFS_RESULTS_ROOT}" \
    --logs-root "${GPFS_LOGS_ROOT}/${RUN_ID}" \
    --arms "${ARMS}" \
    --gpus "${GPUS}" \
    --text-arm "${TEXT_ARM}" \
    --stages "${STAGES}" \
    --expected-gpu-count "${EXPECTED_GPU_COUNT}" \
    --min-free-mem-mib "${MIN_FREE_MEM_MIB}" \
    "${STAGE_ARGS_FLAGS[@]}" \
    "${force_flag[@]}" \
    2>&1 | redact | tee "${controller_log}"
  # PIPESTATUS[0] is h200_pod_exec.sh's own exit code, which is the remote
  # worker's exit code (kubectl exec propagates it) since h200_pod_exec.sh
  # execs kubectl directly rather than wrapping it. `redact` and `tee` are later
  # stages of the same pipeline and do not change that index.
  return "${PIPESTATUS[0]}"
}

# ----------------------------------------------------------------- main

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Required, and checked here rather than at the top of the file so that --help
# works without one. `${H200_POD:?...}` at file scope made the usage text
# unreachable to anyone who did not already know the answer it documents.
if [ -z "${H200_POD:-}" ]; then
  echo "set H200_POD to a running pod name before launching (pods are disposable;" >&2
  echo "this controller does not default to one) -- see ~/hangzhou-remote/README.md" >&2
  exit 2
fi
export H200_POD

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH (needed to render the run manifest)" >&2
  exit 2
fi
require_local_path "project root" "${PROJECT_ROOT}"
require_local_path "src/transfer" "${PROJECT_ROOT}/src/transfer"
require_local_path "scripts/transfer" "${PROJECT_ROOT}/scripts/transfer"
require_local_path "worker script" "${PROJECT_ROOT}/scripts/transfer/h200_worker.sh"
require_local_path "07_convergence_control.py ladder table" \
  "${PROJECT_ROOT}/docs/analysis/MODEL_LADDER_20260728.md"
for tool in "${H200_STATUS_CHECK}" "${H200_SYNC}" "${H200_GPFS_PUSH}" \
  "${H200_POD_BASH}" "${H200_POD_EXEC}"; do
  if [ ! -x "${tool}" ]; then
    echo "required access-layer tool not found or not executable: ${tool}" >&2
    exit 2
  fi
done

# The campaign panel comes from scripts/transfer/panel_contract.sh, generated
# from src/transfer/arms.py's PANEL and sourced by both this controller and the
# worker. It used to be a hand-written string here AND a second hand-written copy
# in h200_worker.sh, either of which could drift from the panel or from the
# other. The worker re-derives the file from the live panel in its preflight, so
# a stale copy cannot reach a measurement; this script only reads it, because the
# freeze step is required to work without torch importable.
KNOWN_ARMS="${TRANSFER_CAMPAIGN_PANEL}"
IFS=',' read -r -a ARM_LIST <<< "${ARMS}"
if [ "${#ARM_LIST[@]}" -eq 0 ]; then
  echo "ARMS must not be empty" >&2
  exit 2
fi
for arm in "${ARM_LIST[@]}"; do
  case " ${KNOWN_ARMS} " in
    *" ${arm} "*) ;;
    *) echo "unknown arm: ${arm} (panel is: ${KNOWN_ARMS})" >&2; exit 2 ;;
  esac
done
reject_duplicate_values ARMS "${ARM_LIST[@]}"
if ! [[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPUS must be a comma-separated list of integers, got: ${GPUS}" >&2
  exit 2
fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
reject_duplicate_values GPUS "${GPU_LIST[@]}"

STAGE_ORDER_CSV="$(IFS=,; echo "${STAGE_NAMES[*]}")"
IFS=',' read -r -a REQUESTED_STAGES <<< "${STAGES}"
if [ "${#REQUESTED_STAGES[@]}" -eq 0 ]; then
  echo "STAGES must not be empty" >&2
  exit 2
fi
for stage in "${REQUESTED_STAGES[@]}"; do
  case " ${STAGE_ORDER_CSV//,/ } " in
    *" ${stage} "*) ;;
    *)
      echo "unknown stage: ${stage} (known stages: ${STAGE_ORDER_CSV})" >&2
      exit 2
      ;;
  esac
done
reject_duplicate_values STAGES "${REQUESTED_STAGES[@]}"
collect_stage_args

SCRATCH="$(mktemp -d)"
trap 'rm -rf -- "${SCRATCH}"' EXIT

log "InterpretabilityTransfer H200 controller"
# The pod name is deliberately absent from this banner and from every other line
# this script writes; see `redact` above. Cluster identity is established by the
# health check, which is the thing that actually needs to be true.
log "PROJECT_ROOT:      ${PROJECT_ROOT}"
log "GPFS_PACKAGE_ROOT: ${GPFS_PACKAGE_ROOT}"
log "GPFS_RESULTS_ROOT: ${GPFS_RESULTS_ROOT}"
log "GPFS_LOGS_ROOT:    ${GPFS_LOGS_ROOT}"
log "ARMS:              ${ARMS}"
log "GPUS:              ${GPUS}"
log "TEXT_ARM:          ${TEXT_ARM}"
log "STAGES:            ${STAGES}"
log "  (full catalog:   ${STAGE_ORDER_CSV})"
log "STAGE_ARGS:        ${STAGE_ARGS_SUMMARY:-(none -- every stage runs at its own script defaults)}"
log "FORCE:             ${FORCE}"
log "DRY_RUN:           ${DRY_RUN}"

log "staging and hashing the complete runtime snapshot"
freeze_manifest "${SCRATCH}"
stage_snapshot "${SCRATCH}"
resolve_run_id
log "run_id=${RUN_ID} code_hash=${CODE_HASH} snapshot_dir=${SNAPSHOT_DIR}"

POD_COMMAND=(
  "${H200_POD_EXEC}" -- bash "${SNAPSHOT_DIR}/scripts/transfer/h200_worker.sh"
  --run-id "${RUN_ID}" --snapshot-dir "${SNAPSHOT_DIR}"
  --results-root "${GPFS_RESULTS_ROOT}" --logs-root "${GPFS_LOGS_ROOT}/${RUN_ID}"
  --arms "${ARMS}" --gpus "${GPUS}" --text-arm "${TEXT_ARM}" --stages "${STAGES}"
  --expected-gpu-count "${EXPECTED_GPU_COUNT}" --min-free-mem-mib "${MIN_FREE_MEM_MIB}"
  "${STAGE_ARGS_FLAGS[@]}"
)
[ "${FORCE}" = "1" ] && POD_COMMAND+=(--force)

if [ "${DRY_RUN}" = "1" ]; then
  log "[dry-run] would run H200 cluster health check: ${H200_STATUS_CHECK}"
  log "[dry-run] would check snapshot absence via: ${H200_POD_BASH} \"test -e '${SNAPSHOT_DIR}' ...\""
  log "[dry-run] would push code snapshot ($( wc -l < "${FILE_LIST}" ) files) via:"
  log "[dry-run]   ${H200_SYNC} push <local-staging> ${SNAPSHOT_DIR}"
  log "[dry-run] would push run manifest via:"
  log "[dry-run]   ${H200_GPFS_PUSH} <local-run-manifest> ${SNAPSHOT_DIR}/INVOCATIONS/<manifest-sha256>.json"
  log "[dry-run] would invoke worker:"
  log "[dry-run]   ${POD_COMMAND[*]}"
  log "dry run complete; nothing was transferred or executed"
  exit 0
fi

verify_h200_cluster
check_snapshot_absence

if [ "${SNAPSHOT_EXISTS}" -eq 1 ]; then
  log "run_id ${RUN_ID} has a checksum-verified snapshot on GPFS; reusing it without re-pushing"
else
  push_snapshot
fi

write_run_manifest "${SCRATCH}"
push_run_manifest

invoke_worker
status=$?
if [ "${status}" -ne 0 ]; then
  echo "worker failed with status ${status} (run_id=${RUN_ID}); see ${CONTROLLER_LOG_DIR}/${RUN_ID}.log" >&2
  exit "${status}"
fi
log "campaign complete: run_id=${RUN_ID} results=${GPFS_RESULTS_ROOT} logs=${GPFS_LOGS_ROOT}/${RUN_ID}"
