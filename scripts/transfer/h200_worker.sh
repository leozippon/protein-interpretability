#!/usr/bin/env bash
set -euo pipefail

# Worker for the H200 transfer-measurement campaign. Runs inside an
# H200 pod, invoked by ../run_transfer_h200.sh (the controller) via
# ~/hangzhou-remote/ssh_tunnel/h200_pod_exec.sh against the frozen code
# snapshot this file ships inside of. Do not invoke this script directly
# from the local L20 host -- it assumes GPFS-mounted models and data.
#
# NEVER DELETE OR RECREATE THE RESULTS ROOT. --results-root is a shared
# GPFS directory that other tracks also write into; `git clean -fdx` run
# inside  (its own git repo, /results/
# gitignored) has already destroyed completed experiment artefacts three
# times, and 04_circuit_primitives.py's own source comments independently
# record the same results-root disappearing mid-run. This script only ever
# creates directories (mkdir -p) and moves individual files into place
# inside them; it never removes or replaces the results root itself, and
# the only `rm -rf` calls in this file target temp directories this
# process itself created a few lines earlier.
#
# Atomicity. Each stage/item writes into a fresh temp directory that is a
# subdirectory of its final output directory (guaranteeing the same
# filesystem, so `mv` is an atomic rename), and only after the underlying
# Python process exits 0 are the produced files moved into place one by
# one. A killed run leaves an orphaned `.tmp.*` directory (safe to delete
# by hand; nothing reads it) but never a partially written file at the
# path a later stage or a human would treat as the real output.
#
# Resume. After a successful move, this script writes
# `<results-root>/<stage>/.manifests/<item>.sha256` in `sha256sum` format
# for exactly the files it just placed. Before running an item, it checks
# whether that manifest already exists and verifies (`sha256sum -c`)
# against the files currently on disk; if that passes, the item is skipped
# unless --force. A manifest that exists but does not verify (e.g. a
# concurrent process touched the results root) is treated as incomplete,
# not as an error, and the item is simply redone.
#
# Import preflight. Before any GPU is touched, verify_entry_points_importable
# imports each selected entry point inside this snapshot (as a module,
# so main() never runs) and fails loudly, listing every failure, if any of
# them cannot be imported. This exists because a code-freeze scope that
# missed src/revision/ once shipped a snapshot that scheduled four GPUs and
# lost all of them two seconds later to a ModuleNotFoundError -- a class of
# failure neither `bash -n` nor the controller's `--dry-run` can catch,
# since neither executes Python.
#
# Dependency order (all twelve contract stages; tier 4 was added when
# 10_homology_control.py and 11_induction_path_patching.py were wired and this
# list was not updated with them, which is the same stale-enumeration class as
# the hand-written import-preflight list below):
#   tier 1  01_cohort_power.py (prerequisite)
#   tier 2  02_pathway_budget.py, then 03_estimand_power.py measure/recommend
#   tier 3  04_circuit_primitives.py, 05_relational_channel.py,
#           06_explanation_channel.py, 07_convergence_control.py,
#           08_lens_family.py, 09_probe_and_erasure.py, in that order
#   tier 4  10_homology_control.py, 11_induction_path_patching.py,
#           14_paa_census.py -- three stages that consume nothing any other
#           stage in this campaign produces, so their order within the tier is
#           free and only the GPU wave shape decides it
#
# Per-stage invocation quirks (read from scripts/transfer/*.py at the time
# this was written; re-verify with --help if a stage script changed):
#   01  --kind {text,protein} --arms A [A ...] --device --with-ec
#       --skip-truncation --dtype --cohort-name --out DIR. Scores every arm
#       passed to one invocation together in one process and writes one
#       combined report, so this worker cannot dispatch it purely per arm.
#       It is split into up to five items instead of the two (text,
#       protein) an by-kind-alone split would give:
#         text                gpt2-large
#         protein_large_vocab protgpt2
#         protein_small_vocab zymctrl (--with-ec; EC-conditioned)
#         protein_progen2_base    progen2-base (default dtype)
#         protein_progen2_medium  progen2-medium (--dtype float32)
#       Two independent reasons force this, both from the port agent's
#       L20-vs-H200 cross-check:
#       (a) `truncation_curve` (src/transfer/budget.py) raises a hard,
#           unhandled RuntimeError for any arm whose vocabulary exceeds
#           1024 when the installed transformers build has no
#           `logits_to_keep` support (true for the pod's 4.52.4, for
#           gpt2-large and ProtGPT2, both vocab 50257). Since 01 writes its
#           combined report only after its whole per-arm loop finishes, one
#           arm raising loses every other arm already computed in that same
#           invocation -- so gpt2-large and ProtGPT2 (vocab > 1024) MUST run
#           with --skip-truncation, and zymctrl/progen2-base/progen2-medium
#           MUST NOT, since they can compute the curve and it is
#           part of the measurement. The guard itself is deliberately not
#           relaxed upstream: trimming is numerically non-inert (up to 0.25
#           in a logit, 0.12 nats in one token's NLL), so a curve computed
#           with it skipped is not comparable to one computed without.
#       (b) progen2-medium's own nll_reduction_shortest_to_longest_nats
#           moved 0.6266 -> 0.7293 (+16%) under bfloat16 in the L20-vs-H200
#           cross-check -- a small difference between two ~2.9-nat
#           endpoints, and the one statistic in the whole cross-check found
#           to be genuinely host-bound. --dtype is one flag for the whole
#           invocation (it governs model loading, so it cannot be set
#           per-arm within one process), so progen2-medium is isolated into
#           its own float32 invocation; that applies float32 to the rest of
#           its cohort_power measurement too, not only the truncation
#           curve, which the cross-check found immaterial for everything
#           else it measured.
#       Each protein sub-item gets an explicit, distinct --cohort-name
#       (default is otherwise "swissprot" for every protein kind
#       invocation regardless of --with-ec/--dtype) so that protgpt2's and
#       progen2-medium's non-EC cohorts -- identical content, hence
#       identical digest under the shared default name -- do not collide on
#       the same output filename.
#       Each truncation curve records `logits_to_keep_used`
#       (src/transfer/budget.py), because whether the trimmed path was taken
#       is host-bound, not arm-bound: ZymCTRL takes the trimmed path on L20
#       and the untrimmed one on the pod. A cross-host comparison of two
#       truncation curves must read that field rather than assume the two
#       hosts agreed.
#   02  --arms A --device --output-root DIR. One JSON per arm; per-arm.
#   03  subcommands `measure` (--arms A --device --output-root DIR) and
#       `recommend` (--arms A [A...] --results-root DIR --output FILE, no
#       --device, CPU-only). measure runs the text arm alone first, then
#       the protein arms (evidence discipline rule 1); recommend runs once
#       afterwards over every arm's already-written measure output.
#   04  --arms A [A ...] --device --output-dir DIR (note: --output-dir, not
#       --output-root). Writes one JSON per arm plus one combined
#       panel_summary.json from the SAME process, so this worker runs 04
#       once with every requested arm together, not per arm -- per-arm
#       invocations would each overwrite panel_summary.json with a
#       different, incomplete panel.
#   05  --arm A (singular) --device --out DIR. Valid only for zymctrl and
#       progen2-medium: the script's own docstring refuses ProtGPT2
#       (multi-residue BPE has no residue-to-token map), and gpt2-large
#       fails the protein-cohort modality check before that.
#   06  no --arm/--arms, no --device, --out DIR. One CPU-only run for the
#       whole panel (Pfam/AlphaFold/Swiss-Prot plus the gpt2-large
#       tokeniser), not per arm.
#   07  no --arm/--arms at all -- it sweeps a --ladder-table of named
#       members (default: every configured member) via optional
#       --members, --device, --output-dir DIR, --backup-dir DIR. This
#       worker does not pass --members (uses the script's own "measure
#       everything configured" default, since the member-name space is not
#       verified to line up with ARMS) and points --backup-dir at pod-local
#       scratch, matching the script's own stated reason for that flag:
#       "results/transfer has twice been deleted by a concurrent
#       process ... the second copy is written under logs/, which is
#       local-only". The default ladder is the code contract in
#       src/transfer/scaling.py; an operator may still pass an explicit
#       --ladder-table on a direct invocation. 07 also has its own graceful
#       per-member availability check
#       (inspect_member) that skips an unstaged rung rather than failing
#       the whole run, so this worker's own data-path preflight for 07
#       stays deliberately narrow -- see verify_item_data_paths.
#   08  --arms A --device --output-root DIR. One JSON per arm; per-arm, and
#       capability-filtered rather than run over the whole arm list -- see
#       LENS_ARMS below. Its own --arms default is sorted(PANEL) with no
#       capability guard, so the arm list is always passed explicitly.
#   09  --arm A (singular) --device --out DIR. Every campaign arm is valid;
#       refusals for a given arm/concept pair are written into the output
#       rather than raised, so no arm restriction is applied here.
#   10  --arms A [A ...] --device --output-dir DIR. Panel-scoped like 04: it
#       builds one homology database and sweeps arms inside one process, so
#       splitting it across GPUs would rebuild that database per shard. Its
#       arm list is the script's OWN PROTEIN_ARMS declaration, mirrored in the
#       panel contract and checked against the source by
#       tests/test_transfer_stage_contract.py.
#   11  --arms A [A ...] --device --output-dir DIR. Panel-scoped for the same
#       reason (one shared repeat cohort). Refuses rotary layouts inside
#       path_patching.require_supported_layout, after the checkpoint is
#       already on the GPU, which is why the contract filters the arm list
#       here instead.
#   14  --census-arm A (singular) --text-arm A --stages S [S ...] --width N
#       --device --out DIR. Per arm: one census plus its causal readout for
#       one arm, against the campaign's declared text control.
#       Three of those flags are fixed here rather than left to
#       ARGS_PAA_CENSUS, each for a reason that is not a preference:
#       (a) --stages census causal. 14_paa_census.py's own default is all
#           five stages, and it REFUSES outright when --census-arm differs
#           from --text-arm while `match` or `query` is requested, because
#           both read the text control's pool and unigram counts and only a
#           census of that arm writes them. Its `gate0` is a panel-wide
#           go/no-go already discharged (EXP-R2-087), not a per-arm
#           measurement. So the default would fail every protein item and
#           measure something else on the text one.
#       (b) --width from the contract's TRANSFER_PAA_CENSUS_WIDTH. This is a
#           feasibility parameter, not a scale knob: prediction_addressed.
#           tokenised_rows keeps only rows reaching EXACTLY the pool width, and
#           in the unchanged 520-800 census band ProtGPT2 admits 320-355 of 400
#           rows at width 192 and 0 at width >= 320 (EXP-R2-082). At the entry
#           point's own default of 512 the arm this stage exists to measure
#           would raise inside tokenised_rows with the checkpoint already on the
#           GPU. The contract's eligible arm list is declared against this
#           width, so it is read from the contract and not written here.
#       (c) --text-arm. The campaign's text control, from --text-arm, so a run
#           cannot anchor on the entry point's own default while the rest of
#           the campaign anchors elsewhere.
#       Everything that only changes the SIZE of the measurement --
#       --census-sequences, --cohort-draw-seed, --census-ban-depth and the
#       --causal-* family -- is left to ARGS_PAA_CENSUS / ARGS_PAA_CENSUS__<ARM>.
#       ZymCTRL is refused by the contract, permanently and with its reason: no
#       pool width admits both its EC-conditioned rendering and ProtGPT2's
#       multi-residue BPE, so it cannot enter a shared-window panel at all.
#
# Neither this worker nor the controller makes any later stage read 01's
# unmeasurable-arms verdict automatically -- 01 only reports it.
#
# DTYPE POLICY: do not pass --dtype unless there is a measured, documented
# reason to override that specific script's own default, and record the
# reason at the call site when you do. Run 20260728160933_83ff09d5a909 lost
# all four `lens_family` items to `FloatingPointError: Jacobian disagrees
# with a central finite difference by 1.0`: build_command forced
# --dtype bfloat16 on every stage, including 08_lens_family.py, whose
# author set --dtype default="float32" deliberately, because lens
# quantities are differences between near-identical distributions and
# bfloat16 rounding is comparable to the effect being measured -- the
# script's own tolerance guard correctly refused to emit a wrong Jacobian
# rather than silently return one. The one remaining override
# (cohort_power's protein_progen2_medium item, --dtype float32) is exactly the
# shape an override should take: narrow, applies to one arm/item only, and
# justified by a specific measured divergence (see that case in
# build_command). Every other stage now runs with no --dtype flag at all,
# letting each script's own default stand, including
# 09_probe_and_erasure.py, whose bfloat16 default happened to match what
# this file used to force -- which is exactly why it was not safe to
# assume the override there was harmless just because it had not yet
# failed.

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_SCRIPTS="${WORKER_DIR}"

# scripts/transfer/panel_contract.sh: the campaign panel, each arm's modality
# and data-location variables, and each stage's eligible arm list with a reason
# for every refusal -- all generated from src/transfer/arms.py by
# panel_contract.py --emit and sourced here rather than restated.
#
# Until 2026-07-29 this file carried its own KNOWN_ARMS string, its own modality
# enumeration, its own lens-arm exclusion and its own relational-arm inclusion,
# and run_transfer_h200.sh carried a second copy of the first of those. Five
# hand-maintained lists, any of which could drift from PANEL or from each other
# without a single downstream number looking wrong -- which is the shape of L18.
# verify_panel_contract below re-derives this file from the live panel before any
# GPU is scheduled, so a stale copy cannot reach a measurement.
PANEL_CONTRACT_SH="${WORKER_DIR}/panel_contract.sh"

TEXT_ARM=""
ARMS=""
GPUS=""
STAGES=""
RUN_ID=""
SNAPSHOT_DIR=""
RESULTS_ROOT=""
LOGS_ROOT=""
EXPECTED_GPU_COUNT=""
MIN_FREE_MEM_MIB="${MIN_FREE_MEM_MIB:-16000}"
FORCE=0
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/tmp/interpretability_transfer_worker}"

#: Failures that must not stop the campaign but must still fail it. Exactly one
#: kind qualifies: an item that produces no input any later stage reads, so
#: continuing cannot build a later result on a broken earlier one. Every entry is
#: printed and the worker exits non-zero at the end.
DEFERRED_FAILURES=()

#: Items that never ran because an input they need was not staged. Skipping them
#: is right -- staging is partial and a later rerun picks them up -- but a
#: campaign that skipped a stage has not measured it, and until EXP-R2-067 the
#: worker logged one SKIP-DATA line and then ended with "campaign complete" and
#: exit 0. That is a false success, and it was not hypothetical:
#: A previous environment omitted the homology inputs, so the default campaign
#: skipped the memorisation control and still reported success. These are listed
#: at the end and the worker exits non-zero, exactly like DEFERRED_FAILURES.
SKIPPED_FOR_DATA=()
SKIP_DATA_STATUS=75

#: Requested stages that produced no measurement at all, with the reason. Same
#: false-success class as SKIPPED_FOR_DATA and closed by the same accounting: a
#: stage that measured nothing must never be indistinguishable from one that
#: succeeded.
#:
#: Two routes reach it, and they were separate holes until EXP-R2-067:
#:
#: 1. A requested stage whose eligible arm list came out empty. run_stage_wave,
#:    run_estimand_power and run_panel_stage each logged one "skipping" line and
#:    returned 0, so `ARMS=gpt2-large,protgpt2 STAGES=relational_channel,
#:    homology_control` -- neither of which can serve either arm -- measured
#:    nothing, printed "campaign complete" and exited 0.
#: 2. A requested stage that no branch of the tier chain below dispatches. The
#:    chain is hand-maintained, one `if stage_requested X` per stage, and nothing
#:    reconciled it against REQUESTED_STAGES: a stage added to the panel contract
#:    and forgotten here passed every preflight (which derives from the contract),
#:    never ran, and exited 0. reconcile_dispatched_stages closes that by
#:    construction -- only dispatch_stage marks a stage as dispatched.
UNMEASURED_STAGES=()

#: Stages dispatch_stage actually handed to a runner, for that reconciliation.
DISPATCHED_STAGES=()

#: Set by finish_campaign so the EXIT trap can tell "the campaign ended and its
#: ledger has been printed" from "the worker exited early and the ledger is about
#: to be lost". See report_early_exit.
CAMPAIGN_LEDGER_PRINTED=0

# Per-stage scale-parameter passthrough (--n-seq, --pool-size, --seeds and so
# on). Populated from repeated --stage-args STAGE BASE64 flags below and
# appended verbatim to that stage's command in build_command. This is how
# the controller's ARGS_<STAGE> environment variables reach the worker; see
# scripts/transfer/README.md's "Controller Environment" for the full table.
declare -A STAGE_EXTRA_ARGS=()
#: Same, scoped to one item of one stage, keyed "stage/item". A stage-wide flag
#: cannot be overridden for a single item, and cohort_power's four items differ
#: in exactly the ways that make one scale knob wrong for the others.
declare -A ITEM_EXTRA_ARGS=()

# ------------------------------------------------------------------ helpers

# The valid STAGE and ITEM values are deliberately NOT enumerated here. This
# text used to name nine stages while the worker accepted and dispatched eleven
# -- a hand-maintained list that drifted from the contract exactly the way the
# five lists panel_contract.sh replaced did, and one an operator reading --help
# would have believed. The authority is TRANSFER_STAGE_ORDER in
# scripts/transfer/panel_contract.sh, which the argument validation below checks
# against and which --help cannot print because the contract is sourced only
# after the snapshot manifest is verified (and that needs these arguments).
usage() {
  cat <<'EOF'
Usage: h200_worker.sh --run-id ID --snapshot-dir DIR --results-root DIR
         --logs-root DIR --arms A,A,... --gpus N,N,... --text-arm ARM
         --stages STAGE,STAGE,...
         [--expected-gpu-count N] [--min-free-mem-mib N]
         [--stage-args STAGE BASE64 ...] [--item-args STAGE ITEM BASE64 ...]
         [--force]

--expected-gpu-count, if given, is an extra minimum-count assertion on top
of whatever nvidia-smi reports; the default is to trust nvidia-smi alone,
since pods are disposable and the GPU count varies between them.

--stage-args STAGE BASE64 appends the base64-decoded, space-split argument
string to every item of that stage's invocation (e.g. "--n-seq 500
--pool-size 1000"). Repeatable, once per stage that needs an override.

--item-args STAGE ITEM BASE64 is the same scoped to one item of one stage,
because a stage-wide knob cannot express "give the residue cohort_power item
a different --n-seq" without moving the other items with it. Repeatable, once
per stage/item pair.

Either kind is refused if it repeats a flag this worker already sets for that
item, rather than letting argparse silently take the last occurrence.

Valid STAGE values are TRANSFER_STAGE_ORDER in
scripts/transfer/panel_contract.sh (generated from src/transfer/arms.py); an
unknown stage is refused at argument validation. ITEM is the stage's own item
namespace: an arm name for a per-arm stage, one of TRANSFER_COHORT_ITEMS for
cohort_power, and the literal "panel" for a panel-wide stage.
EOF
}

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
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

# ----------------------------------------------------------------- arguments

while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --snapshot-dir) SNAPSHOT_DIR="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --logs-root) LOGS_ROOT="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --text-arm) TEXT_ARM="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --expected-gpu-count) EXPECTED_GPU_COUNT="$2"; shift 2 ;;
    --min-free-mem-mib) MIN_FREE_MEM_MIB="$2"; shift 2 ;;
    --stage-args)
      decoded="$(printf '%s' "$3" | base64 -d 2>/dev/null)" || {
        echo "failed to base64-decode --stage-args for stage $2" >&2
        exit 2
      }
      if [ -n "${STAGE_EXTRA_ARGS[$2]+x}" ]; then
        echo "duplicate --stage-args scope: $2" >&2
        exit 2
      fi
      STAGE_EXTRA_ARGS["$2"]="${decoded}"
      shift 3
      ;;
    --item-args)
      decoded="$(printf '%s' "$4" | base64 -d 2>/dev/null)" || {
        echo "failed to base64-decode --item-args for stage $2 item $3" >&2
        exit 2
      }
      if [ -n "${ITEM_EXTRA_ARGS[$2/$3]+x}" ]; then
        echo "duplicate --item-args scope: $2/$3" >&2
        exit 2
      fi
      ITEM_EXTRA_ARGS["$2/$3"]="${decoded}"
      shift 4
      ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in RUN_ID SNAPSHOT_DIR RESULTS_ROOT LOGS_ROOT ARMS GPUS TEXT_ARM STAGES; do
  if [ -z "${!required}" ]; then
    echo "missing required argument: ${required} was not set" >&2
    usage >&2
    exit 2
  fi
done

# The run-id's own trailing segment is the code hash the controller minted
# it against (see run_transfer_h200.sh's resolve_run_id); reused here as the
# provenance record's code-identity token rather than passed as a separate
# flag, so there is exactly one source of truth for "which code produced
# this".
CODE_HASH_SHORT="${RUN_ID##*_}"

verify_snapshot_manifest() {
  local manifest="${SNAPSHOT_DIR}/CODE_CONTENT_SHA256SUMS"
  local manifest_hash
  if [ ! -f "${manifest}" ]; then
    echo "snapshot manifest is missing: ${manifest}" >&2
    exit 2
  fi
  manifest_hash="$(sha256sum "${manifest}" | awk '{print $1}')"
  if [ "${manifest_hash:0:12}" != "${CODE_HASH_SHORT}" ]; then
    echo "snapshot manifest hash does not match run-id ${RUN_ID}" >&2
    exit 2
  fi
  if ! (cd "${SNAPSHOT_DIR}" && sha256sum -c -- CODE_CONTENT_SHA256SUMS >/dev/null); then
    echo "snapshot content checksum verification failed: ${SNAPSHOT_DIR}" >&2
    exit 2
  fi
}

verify_snapshot_manifest

if [ ! -f "${PANEL_CONTRACT_SH}" ]; then
  echo "missing ${PANEL_CONTRACT_SH} -- run" >&2
  echo "  python scripts/transfer/panel_contract.py --emit" >&2
  echo "and re-freeze; the worker will not schedule without the panel contract" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${PANEL_CONTRACT_SH}"

# ------------------------------------------------------------- pod environment

# scripts/transfer/h200_env.sh: GPFS path overrides for this pod, written by
# the port agent, sourced here rather than duplicated. It exports (among
# other things) TRANSFER_REPO_ROOT, TRANSFER_MODEL_BASE_DIR, TRANSFER_TEXT_MODEL_DIR,
# TRANSFER_TEXT_MODEL_BASE_DIR, TRANSFER_OPENWEBTEXT_DIR, TRANSFER_SWISSPROT_FASTA,
# TRANSFER_ZYMCTRL_FASTA, TRANSFER_PFAM_RESIDUE_TSV, TRANSFER_PROTEINGYM_DIR, TRANSFER_ALPHAFOLD_DIR
# (the variables src/transfer/*.py actually reads via env_path/
# require_input_path), plus TRANSFER_PYTHON (the pod's interpreter -- there is no
# conda env here) and TRANSFER_PACKAGE_ROOT (feeds PYTHONPATH so `import
# src.transfer...` resolves). h200_env.sh's internal roots are not read here
# directly. TRANSFER_PACKAGE_ROOT is exported *before* sourcing so h200_env.sh's
# own `${TRANSFER_PACKAGE_ROOT:-default}` picks up this run's own immutable
# snapshot rather than its generic, run-id-less default -- otherwise
# `import src.transfer` could resolve to a different run's code.
if [ ! -f "${WORKER_DIR}/h200_env.sh" ]; then
  echo "missing ${WORKER_DIR}/h200_env.sh -- the pod-environment file this worker" >&2
  echo "depends on has not been added to the snapshot yet" >&2
  exit 2
fi
export TRANSFER_PACKAGE_ROOT="${SNAPSHOT_DIR}"
# shellcheck source=/dev/null
source "${WORKER_DIR}/h200_env.sh"

if [ -z "${TRANSFER_PYTHON:-}" ] || [ ! -x "${TRANSFER_PYTHON}" ]; then
  echo "h200_env.sh did not export a usable TRANSFER_PYTHON interpreter (got: '${TRANSFER_PYTHON:-}')" >&2
  exit 2
fi

# --------------------------------------------------------------- preflight

# Run 20260728150714_b613d3afe620 scheduled four GPUs and lost all of them
# two seconds later to `ModuleNotFoundError: No module named 'src.revision'`
# -- a missing dependency that neither `bash -n` nor the controller's
# `--dry-run` could ever catch, since neither executes Python. This runs
# before any GPU is touched: it imports each entry point this run will
# actually invoke (as a module, not as __main__, so main() never runs and no
# model or GPU work happens) inside the frozen snapshot and fails loudly,
# collecting every failure rather than stopping at the first, if any
# import raises -- whether that is a missing dependency (what this exists
# because of) or a syntax error in a file still being edited (which is a
# real defect and must fail here too, not be worked around).
#
# Each entry point gets its OWN short-lived interpreter (one `TRANSFER_PYTHON -c`
# subprocess per file), not one shared interpreter for all of them. Run
# 20260728152900_02f91a55c9e7 hit a false positive from the shared-
# interpreter version: `03_estimand_power.py` failed with
# `AttributeError: 'NoneType' object has no attribute '__dict__'` under the
# preflight while importing cleanly standalone, on both the pod and L20 --
# the classic signature of an earlier entry point's import leaving a `None`
# negative-cache marker in `sys.modules` that a later, unrelated entry
# point's import then tripped over. Subprocess isolation removes the
# shared `sys.modules` entirely, so one entry point's import state can
# never leak into another's -- this makes the check test what it claims to
# test ("does this file import in a clean interpreter"), not "does it
# import after eight others have already been imported into the same
# one". No self-test reproduces the contamination class: the fault is
# visible only across two entry points sharing one interpreter, which
# subprocess isolation now makes unconstructible, and a check pointing at
# a function that does not exist was the only thing recording that. The
# cost of the isolation is one
# python/torch/transformers startup per entry point instead of one shared
# process; still seconds, not minutes, and far cheaper than a GPU
# scheduled against a false negative.
import_one_entry_point() {
  local path="$1"
  "${TRANSFER_PYTHON}" -c '
import importlib.util
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("_import_preflight_entry_point", path)
module = importlib.util.module_from_spec(spec)
# Registered before exec, per importlib recommended practice: some library
# code (e.g. dataclasses/typing resolving forward-referenced annotations)
# looks itself up via sys.modules[__name__] during its own execution, and an
# unregistered name is exactly the shape of state this preflight exists to
# stop being ambiguous with a genuine failure.
sys.modules[spec.name] = module
spec.loader.exec_module(module)
' "${path}"
}

# The entry points this run will actually invoke, from the panel contract.
#
# This used to be a hand-written list of nine, while the worker schedules eleven
# stages: 10_homology_control.py and 11_induction_path_patching.py were dispatched
# without ever being import-checked, which is exactly the class of failure this
# preflight exists to stop -- a code-freeze scope gap that loses four GPUs two
# seconds in. Deriving it from TRANSFER_STAGE_ENTRY means a new stage is covered by
# construction.
requested_entry_points() {
  local stage
  for stage in "${REQUESTED_STAGES[@]}"; do
    printf '%s\n' "${TRANSFER_STAGE_ENTRY[${stage}]}"
  done | LC_ALL=C sort -u
}

verify_entry_points_importable() {
  local -a scripts=()
  local script path output status
  local -a failures=()
  while IFS= read -r script; do scripts+=("${script}"); done < <(requested_entry_points)

  log "import preflight: checking ${#scripts[@]} scheduled entry point(s) import cleanly, one subprocess each"
  for script in "${scripts[@]}"; do
    path="${TRANSFER_SCRIPTS}/${script}"
    if [ ! -f "${path}" ]; then
      failures+=("${path}: file does not exist in the snapshot")
      continue
    fi
    status=0
    output="$(import_one_entry_point "${path}" 2>&1)" || status=$?
    if [ "${status}" -ne 0 ]; then
      failures+=("${path}: exit ${status}: ${output}")
    else
      log "  ok: ${path}"
    fi
  done

  if [ "${#failures[@]}" -gt 0 ]; then
    echo "import preflight failed for one or more entry points:" >&2
    local f
    for f in "${failures[@]}"; do
      echo "  FAIL: ${f}" >&2
    done
    echo "No GPU has been scheduled. Fix the frozen snapshot (a code-freeze" >&2
    echo "scope gap) or the code itself (a real defect) and relaunch." >&2
    exit 2
  fi
  log "import preflight passed for every scheduled entry point"
}

# Argparse construction (the `parser = ArgumentParser(); parser.add_argument(...)`
# calls) happens inside main()/parse_args(), not at module import time, so
# verify_entry_points_importable's bare import never exercises it. `--help`
# does: argparse builds the whole parser and then exits 0 on its own,
# before main() reaches any real work, so this is as cheap and as side-
# effect-free as the import check. Offered because it is cheap, not
# because it is comprehensive: it would NOT have caught the call-signature
# drift the port agent found and fixed by hand in
# 07_convergence_control.py (protein_repeat_cohort/text_repeat_cohort
# taking a RepeatCriterion value where circuits.py used to accept a loose
# min_unit keyword) -- that is a runtime call inside a function body,
# reached only once real measurement code executes past argument parsing,
# not an argparse-construction-time error, and nothing here claims to
# catch that class.
verify_entry_points_parse_args() {
  local -a scripts=()
  local script path output status sub label
  local -a failures=() subs
  while IFS= read -r script; do scripts+=("${script}"); done < <(requested_entry_points)

  log "help preflight: exercising argparse construction for ${#scripts[@]} scheduled entry point(s)"
  for script in "${scripts[@]}"; do
    path="${TRANSFER_SCRIPTS}/${script}"
    [ -f "${path}" ] || continue  # already reported by the import preflight
    case "${script}" in
      03_estimand_power.py) subs=(measure recommend) ;;  # required subcommands
      *) subs=("") ;;
    esac
    for sub in "${subs[@]}"; do
      label="${path}${sub:+ ${sub}} --help"
      status=0
      if [ -n "${sub}" ]; then
        output="$("${TRANSFER_PYTHON}" "${path}" "${sub}" --help 2>&1)" || status=$?
      else
        output="$("${TRANSFER_PYTHON}" "${path}" --help 2>&1)" || status=$?
      fi
      if [ "${status}" -ne 0 ]; then
        failures+=("${label}: exit ${status}: ${output}")
      else
        log "  ok: ${label}"
      fi
    done
  done

  if [ "${#failures[@]}" -gt 0 ]; then
    echo "help preflight failed for one or more entry points:" >&2
    local f
    for f in "${failures[@]}"; do
      echo "  FAIL: ${f}" >&2
    done
    echo "No GPU has been scheduled." >&2
    exit 2
  fi
  log "help preflight passed for every scheduled entry point"
}

verify_gpfs_read_write() {
  local probe="${RESULTS_ROOT}/.gpfs_rw_probe.${RUN_ID}"
  local fs_type
  mkdir -p "${RESULTS_ROOT}"
  fs_type="$(stat -f -c %T "${RESULTS_ROOT}" 2>/dev/null || echo unknown)"
  if [ "${fs_type}" != "gpfs" ]; then
    echo "results root is not on a gpfs filesystem (stat -f reports: ${fs_type}): ${RESULTS_ROOT}" >&2
    exit 2
  fi
  if ! { echo probe > "${probe}" 2>/dev/null && [ "$(cat -- "${probe}" 2>/dev/null)" = probe ]; }; then
    echo "GPFS results root is not writable: ${RESULTS_ROOT}" >&2
    exit 2
  fi
  rm -f -- "${probe}"
  log "GPFS results root verified read-write and on gpfs: ${RESULTS_ROOT}"
}

# h200_env.sh deliberately does not check that its paths exist: data/swissprot
# and data/alphafold are still being staged, and a measurement needing
# neither must still be able to run. So this worker checks only the
# variables the item about to run actually needs, right before that item
# runs, and leaves anything else to src.transfer.arms.require_input_path,
# which already names the offending variable in its failure message. A
# blanket all-ten-paths preflight would block runs that are legitimately
# possible today.

# Modality, checkpoint variable and corpus variables all come from the panel
# contract (see the source at the top of this file). Every one of the three used
# to be a `case` on the arm's name here, and the modality one was silently wrong
# for the eleven-arm panel until 2026-07-29: it treated "gpt2-large" as the only
# text arm, so six text arms would have been handed the Swiss-Prot corpus
# variable and the protein model root. An unknown arm now fails loudly on the
# associative-array lookup instead of falling through to a default, which is the
# property that was missing.
arm_modality() {
  local value="${TRANSFER_ARM_MODALITY[$1]:-}"
  if [ -z "${value}" ]; then
    echo "arm_modality: $1 is not in the panel contract (TRANSFER_CAMPAIGN_PANEL=${TRANSFER_CAMPAIGN_PANEL})" >&2
    exit 2
  fi
  printf '%s\n' "${value}"
}

model_var_for_arm() {
  local value="${TRANSFER_ARM_MODEL_VAR[$1]:-}"
  if [ -z "${value}" ]; then
    echo "model_var_for_arm: $1 is not in the panel contract" >&2
    exit 2
  fi
  printf '%s\n' "${value}"
}

# The arm's own checkpoint directory, not the root the variable above points at.
#
# The variable is the wrong granularity for a preflight and it mattered: six of
# the seven text arms resolve TRANSFER_TEXT_MODEL_BASE_DIR, which is the models
# ROOT. That directory exists as soon as any text checkpoint is staged, so an arm
# whose own checkpoint was absent passed this check and raised inside load_arm
# instead -- and cohort_power scores all seven text arms in ONE process, so the
# one missing checkpoint took the six that were fine down with it, mid-run,
# instead of the item being reported as a skip before anything was scheduled.
#
# The relative segment comes from the contract's TRANSFER_ARM_MODEL_REL
# (panel_contract.py::model_relative_path, which reads ArmSpec.path), so no leaf
# name is written here and re-pointing any of the three environment variables
# moves the root and the checkpoint together. "." means the arm IS the variable
# (gpt2-large is declared as TRANSFER_TEXT_MODEL_DIR itself).
model_path_for_arm() {
  local arm="$1" var rel root
  var="$(model_var_for_arm "${arm}")"
  if [ -z "${TRANSFER_ARM_MODEL_REL[${arm}]+set}" ]; then
    echo "model_path_for_arm: ${arm} has no checkpoint path in the panel contract" >&2
    exit 2
  fi
  rel="${TRANSFER_ARM_MODEL_REL[${arm}]}"
  root="${!var:-}"
  if [ -z "${root}" ]; then
    return 1
  fi
  if [ "${rel}" = "." ]; then
    printf '%s\n' "${root}"
  else
    printf '%s\n' "${root}/${rel}"
  fi
}

# Corpus variables needed to build a cohort covering the given arms.
corpus_vars_for_arms() {
  local arm var
  for arm in "$@"; do
    if [ -z "${TRANSFER_ARM_CORPUS_VARS[${arm}]+set}" ]; then
      echo "corpus_vars_for_arms: ${arm} is not in the panel contract" >&2
      exit 2
    fi
    for var in ${TRANSFER_ARM_CORPUS_VARS[${arm}]}; do
      printf '%s\n' "${var}"
    done
  done
  return 0
}

# The arms a stage may actually run, from the panel contract's arm_can_run
# predicate, with every refusal logged. A stage that is handed an arm its entry
# point cannot serve does not fail cheaply: 11_induction_path_patching.py on a
# rotary arm raises inside path_patching.require_supported_layout, and on a
# ProGen2 arm inside attention_output_projection, both after the checkpoint is on
# the GPU and the repeat cohort has been scanned. Filtering here, with the
# reason, is what turns that into a logged skip.
stage_eligible_arms() {
  local stage="$1"
  shift
  local -a requested=("$@")
  local arm allowed reason
  STAGE_ARMS_OUT=()
  if [ -z "${TRANSFER_STAGE_ARMS[${stage}]+set}" ]; then
    echo "stage_eligible_arms: ${stage} is not in the panel contract" >&2
    exit 2
  fi
  allowed=" ${TRANSFER_STAGE_ARMS[${stage}]} "
  for arm in "${requested[@]}"; do
    case "${allowed}" in
      *" ${arm} "*) STAGE_ARMS_OUT+=("${arm}") ;;
      *)
        reason="${TRANSFER_STAGE_REFUSAL[${stage}/${arm}]:-not eligible (no reason recorded)}"
        log "SKIP-ARM   stage=${stage} arm=${arm} (${reason})"
        ;;
    esac
  done
}

# The arm(s) a given (stage, item) pair actually touches, for data-path
# scoping.
#
# Dispatch is on the stage's DECLARED SCOPE, not on its name. Naming the stages
# one by one is how this file previously decided the same question in
# verify_commands_buildable, where a stage absent from the list fell through to
# a catch-all and was built with the literal item "panel" as though it were an
# arm; that was repaired to read the contract and this function was left behind
# with the identical shape. Under a name list a newly declared panel-wide stage
# lands in the per-arm branch, and the data-path check for it then resolves
# model variables for an arm called "panel" -- a preflight that passes because
# it checked nothing. The scope is declared once, in panel_contract.py, and is
# read here.
#
# cohort_power stays a named case, and is the only one: it is panel-wide yet its
# item space is neither "panel" nor an arm name but the four cohort labels the
# contract declares in TRANSFER_COHORT_ITEM_ARMS, which is where its arm lists
# come from. An unknown scope is refused rather than guessed.
arms_for_item() {
  local stage="$1" item="$2" arm
  if [ "${stage}" = cohort_power ]; then
    for arm in ${COHORT_ITEM_ARMS_FOR[${item}]:-}; do printf '%s\n' "${arm}"; done
    return 0
  fi
  if [ -z "${TRANSFER_STAGE_SCOPE[${stage}]:-}" ]; then
    echo "stage ${stage} has no declared scope in the panel contract; this worker cannot resolve its arms" >&2
    exit 2
  fi
  case "${TRANSFER_STAGE_SCOPE[${stage}]}" in
    armless) : ;;
    per_arm|control_anchored) printf '%s\n' "${item}" ;;
    *)
      # Panel-wide: the item is the literal "panel" and the arms are the
      # stage's own contract-eligible list, the same list run_panel_stage
      # passes to the entry point.
      for arm in ${STAGE_ARMS_FOR[${stage}]:-}; do printf '%s\n' "${arm}"; done
      ;;
  esac
}

# Variables this worker is confident a stage needs beyond the arm-derived
# model/corpus variables above. Deliberately conservative for 07 and 09:
# their real data dependency is conditional on data this worker cannot
# inspect from bash (which ladder rungs are locally staged for 07; which
# probe concepts a run reaches for 09), so only what is directly confirmed
# from source is checked here, and the rest is left to
# require_input_path -- consistent with not blocking a run that might not
# need the missing path at all.
extra_vars_for_stage() {
  case "$1" in
    relational_channel) echo TRANSFER_ALPHAFOLD_DIR; echo TRANSFER_PFAM_RESIDUE_TSV ;;
    explanation_channel)
      echo TRANSFER_TEXT_MODEL_DIR
      echo TRANSFER_OPENWEBTEXT_DIR
      echo TRANSFER_SWISSPROT_FASTA
      echo TRANSFER_ALPHAFOLD_DIR
      echo TRANSFER_PFAM_RESIDUE_TSV
      ;;
    convergence_control)
      echo TRANSFER_MODEL_BASE_DIR
      echo TRANSFER_TEXT_MODEL_DIR
      echo TRANSFER_TEXT_MODEL_BASE_DIR
      ;;
    probe_and_erasure) echo TRANSFER_PROTEINGYM_DIR ;;
    homology_control)
      # Database, extraction and scratch paths are outputs created by the stage.
      echo TRANSFER_UNIREF50_FASTA
      echo TRANSFER_DIAMOND_TARBALL
      echo TRANSFER_DIAMOND_CHECKSUM
      ;;
  esac
}

# Absent data is a scheduling fact, not a defect: a missing input path
# means this item cannot run yet (staging is partial and this worker has
# no way to know when that changes), so this returns 1 with
# MISSING_DATA_REASON set for the caller to log as a skip, rather than
# exiting the process the way a genuine computation error does. Skipped
# items write no manifest, so a later run of the same command retries them
# once the input lands -- see run_item_atomic and "Resume And Output Safety"
# in scripts/transfer/README.md.
verify_item_data_paths() {
  local stage="$1" item="$2"
  local -a item_arms=() vars=() checked=() checkpoints=()
  local a v value already c checkpoint status

  MISSING_DATA_REASON=""
  while IFS= read -r a; do
    [ -n "${a}" ] && item_arms+=("${a}")
  done < <(arms_for_item "${stage}" "${item}")

  # Checkpoints first, at checkpoint granularity rather than at the granularity of
  # the variable that relocates them -- see model_path_for_arm for why the
  # variable alone let a missing text checkpoint through and lost six arms with
  # it. The variable's own absence is reported separately from the checkpoint's,
  # because "h200_env.sh exports nothing for this arm" and "the root is staged but
  # this arm's checkpoint is not in it" are different scheduling facts.
  for a in "${item_arms[@]}"; do
    status=0
    checkpoint="$(model_path_for_arm "${a}")" || status=$?
    case "${status}" in
      0) ;;
      1)
        MISSING_DATA_REASON="h200_env.sh did not export $(model_var_for_arm "${a}"), which arm ${a} needs"
        return 1
        ;;
      *)
        # model_path_for_arm named the problem on stderr already. An arm the
        # contract does not know is a defect, not a staging fact, so it must not
        # be laundered into a skip.
        exit "${status}"
        ;;
    esac
    if [ ! -e "${checkpoint}" ]; then
      MISSING_DATA_REASON="missing checkpoint for arm ${a}: ${checkpoint}"
      return 1
    fi
    checkpoints+=("${checkpoint}")
  done

  if [ "${#item_arms[@]}" -gt 0 ]; then
    while IFS= read -r v; do
      [ -n "${v}" ] && vars+=("${v}")
    done < <(corpus_vars_for_arms "${item_arms[@]}")
  fi
  while IFS= read -r v; do
    [ -n "${v}" ] && vars+=("${v}")
  done < <(extra_vars_for_stage "${stage}")

  for v in "${vars[@]}"; do
    already=0
    for c in "${checked[@]}"; do
      [ "${c}" = "${v}" ] && already=1
    done
    [ "${already}" = 1 ] && continue
    checked+=("${v}")
    value="${!v:-}"
    if [ -z "${value}" ]; then
      MISSING_DATA_REASON="h200_env.sh did not export ${v}, which this stage needs"
      return 1
    fi
    if [ ! -e "${value}" ]; then
      MISSING_DATA_REASON="missing input for \$${v}: ${value}"
      return 1
    fi
  done
  if [ "${#checkpoints[@]}" -gt 0 ] || [ "${#checked[@]}" -gt 0 ]; then
    log "  data paths ok for ${stage}/${item}: checkpoints=[${checkpoints[*]:-}] vars=[${checked[*]:-}]"
  fi
  return 0
}

gpu_free_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits
}

# Standing rule 13's before-and-after GPU and memory record.
#
# Both call sites used to end in `|| true`, which is precisely the mechanism by
# which the record the rule requires can be silently absent: an nvidia-smi that
# failed emitted nothing and said nothing, and a reader of the log could not tell
# "the GPUs were idle" from "nobody looked". A failure is logged as a failure now.
# It still does not stop the campaign -- this is evidence about the host, not an
# input to any measurement, and verify_gpus has already refused an occupied or
# invisible GPU by the time the first call runs.
record_host_state() {
  local label="$1" status=0
  log "host state ${label} (standing rule 13):"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv || status=$?
  if [ "${status}" -ne 0 ]; then
    log "WARNING: nvidia-smi failed with status ${status} ${label}; standing rule 13's GPU record is MISSING for this run"
  fi
  status=0
  free -h || status=$?
  if [ "${status}" -ne 0 ]; then
    log "WARNING: free -h failed with status ${status} ${label}; standing rule 13's memory record is MISSING for this run"
  fi
}

# Occupancy (another process already using the GPU) is a hard failure --
# scheduling onto a busy GPU is simply wrong. Free memory is a soft
# warning, not a gate: MIN_FREE_MEM_MIB is not a measured requirement for
# any of the entry-point scripts (observed L20 validation peaks were 1.6-12.2 GiB
# per arm), and a fabricated threshold that blocks a valid run is worse
# than no threshold.
verify_gpu_idle() {
  local gpu="$1" processes free_mib
  processes="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if grep -Eq '[0-9]' <<<"${processes}"; then
    echo "GPU ${gpu} is occupied; refusing to schedule" >&2
    exit 2
  fi
  if [ -n "${MIN_FREE_MEM_MIB}" ]; then
    free_mib="$(gpu_free_mib "${gpu}")"
    if [ "${free_mib}" -lt "${MIN_FREE_MEM_MIB}" ]; then
      log "WARNING: GPU ${gpu} has only ${free_mib} MiB free (< MIN_FREE_MEM_MIB=${MIN_FREE_MEM_MIB}); continuing -- this threshold is unprofiled, not a hard requirement"
    fi
  fi
}

# GPU count is derived from nvidia-smi, not hard-coded: pods are disposable
# and the next one may not have the same count as the one this was written
# against (4 idle H200s, 143 GB each). --expected-gpu-count is an optional
# extra assertion, not the source of truth.
verify_gpus() {
  local visible_count gpu
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found on PATH" >&2
    exit 2
  fi
  visible_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  log "nvidia-smi reports ${visible_count} visible GPU(s)"
  if [ -n "${EXPECTED_GPU_COUNT}" ] && [ "${visible_count}" -lt "${EXPECTED_GPU_COUNT}" ]; then
    echo "expected at least ${EXPECTED_GPU_COUNT} GPUs (explicitly requested), nvidia-smi reports ${visible_count}" >&2
    exit 2
  fi
  for gpu in "${GPU_LIST[@]}"; do
    if [ "${gpu}" -ge "${visible_count}" ]; then
      echo "requested GPU index ${gpu} is not visible; nvidia-smi reports ${visible_count} GPU(s) (0..$((visible_count - 1)))" >&2
      exit 2
    fi
    verify_gpu_idle "${gpu}"
    log "GPU ${gpu} idle and ready"
  done
  if ! "${TRANSFER_PYTHON}" -c 'import torch, transformers' >/dev/null 2>&1; then
    echo "${TRANSFER_PYTHON} cannot import torch/transformers; h200_env.sh did not fully set up the pod's python environment" >&2
    exit 2
  fi
  log "GPU visibility/idle and python environment verified"
}

# ------------------------------------------------------------- stage runner

stage_final_dir() {
  echo "${RESULTS_ROOT}/$1"
}

# Resume must be keyed on provenance, not only on file integrity. The
# results root is shared across runs and outlives any one of them: a
# validation-scale run's manifest checksums verify just as cleanly as a
# production-scale run's, so checksum verification alone cannot tell a
# smaller, stale configuration from the one actually being asked for. This
# programme has retracted conclusions to exactly this kind of silent
# measurement difference before. So every completed item also gets a
# `.provenance` record -- this run's code hash plus its full argument
# vector, with only the GPU index and the instance-specific output path
# normalized out, since neither bears on what was measured -- and a skip
# requires that record to match the current run's, not just a clean
# checksum.
#
# Accepted unresolved defect: model and corpus trees are still identified by
# mutable paths rather than content digests. Input-tree identity needs a
# broader data/checkpoint manifest contract; path hashing would be false safety.

# Prints CMD (as passed in "$@") with the value following --device, --out,
# --output-root, --output-dir, --backup-dir, --results-root or --output
# replaced by a fixed placeholder.
canonicalize_command() {
  local -a out=()
  local -a args=("$@")
  local i=0 n="$#"
  while [ "${i}" -lt "${n}" ]; do
    case "${args[$i]}" in
      --device|--out|--output-root|--output-dir|--backup-dir|--results-root|--output)
        out+=("${args[$i]}" "<normalized>")
        i=$(( i + 2 ))
        ;;
      *)
        out+=("${args[$i]}")
        i=$(( i + 1 ))
        ;;
    esac
  done
  printf '%s\n' "${out[*]}"
}

# The single-line provenance string for one invocation: this run's code hash
# plus its canonical argument vector.
provenance_record() {
  printf 'code_hash=%s command=%s\n' "${CODE_HASH_SHORT}" "$(canonicalize_command "$@")"
}

# Complete means: a checksum manifest exists and verifies against the files
# on disk, AND a provenance record exists and exactly matches the one this
# run would produce. Either failing means "not complete" -- redo, not skip.
item_is_complete() {
  local stage="$1" item="$2" expected_provenance="$3" final_dir stored
  final_dir="$(stage_final_dir "${stage}")"
  [ -f "${final_dir}/.manifests/${item}.sha256" ] || return 1
  [ -f "${final_dir}/.manifests/${item}.provenance" ] || return 1
  stored="$(cat -- "${final_dir}/.manifests/${item}.provenance")"
  [ "${stored}" = "${expected_provenance}" ] || return 1
  ( cd "${final_dir}" && sha256sum -c --quiet -- ".manifests/${item}.sha256" ) >/dev/null 2>&1
}

# Human-readable reason an item is not complete, for the "redo" log line --
# an operator must be able to see what was reused (or why not) rather than
# infer it.
explain_incomplete() {
  local stage="$1" item="$2" expected_provenance="$3" final_dir stored
  final_dir="$(stage_final_dir "${stage}")"
  if [ ! -f "${final_dir}/.manifests/${item}.sha256" ]; then
    echo "no prior output"
    return
  fi
  if [ ! -f "${final_dir}/.manifests/${item}.provenance" ]; then
    echo "prior output has no recorded provenance (pre-dates this check)"
    return
  fi
  stored="$(cat -- "${final_dir}/.manifests/${item}.provenance")"
  if [ "${stored}" != "${expected_provenance}" ]; then
    echo "recorded provenance does not match this run -- stored: [${stored}] current: [${expected_provenance}]"
    return
  fi
  if ! ( cd "${final_dir}" && sha256sum -c --quiet -- ".manifests/${item}.sha256" ) >/dev/null 2>&1; then
    echo "provenance matches but checksum verification failed against files currently on disk"
    return
  fi
  echo "unknown"
}

# Populates the global CMD array for one (stage, item) pair. `item` is an
# arm name for every per-arm stage, a kind (text|protein) for cohort_power,
# and the fixed literal "panel" for the whole-panel single-job stages
# (circuit_primitives, explanation_channel, convergence_control), which
# ignore it and read ARM_LIST/config directly.
build_command() {
  local stage="$1" item="$2" gpu="$3" out_dir="$4"
  CMD=("${TRANSFER_PYTHON}")
  case "${stage}" in
    cohort_power)
      # See the "01" entry in the header comment for why this is four items,
      # not two: the vocab>1024/logits_to_keep truncation-curve guard and
      # progen2-medium's host-bound truncation statistic both force it, and
      # each protein sub-item needs its own --cohort-name so that identical
      # non-EC cohorts (same digest under the shared default name) do not
      # collide on the same output filename.
      # Item arms, extra flags (--skip-truncation / --with-ec / --dtype float32)
      # and --cohort-name all come from panel_contract.py::cohort_power_items,
      # where each carries its measured reason. The float32 override on the
      # residue item is the one deliberate --dtype in this file:
      # progen2-medium's nll_reduction_shortest_to_longest_nats moved
      # 0.6266 -> 0.7293 (+16%) under bfloat16 in the L20-vs-H200 cross-check and
      # collapsed to 2.6e-7 in float32.
      if [ -z "${COHORT_ITEM_ARMS_FOR[${item}]+set}" ]; then
        echo "cohort_power: unknown item ${item}" >&2
        exit 2
      fi
      local -a item_arms=() item_extra=()
      read -r -a item_arms <<< "${COHORT_ITEM_ARMS_FOR[${item}]}"
      read -r -a item_extra <<< "${TRANSFER_COHORT_ITEM_ARGS[${item}]}"
      CMD+=("${TRANSFER_SCRIPTS}/01_cohort_power.py"
            --kind "$(arm_modality "${item_arms[0]}")"
            --arms "${item_arms[@]}"
            --device "cuda:${gpu}")
      [ "${#item_extra[@]}" -gt 0 ] && CMD+=("${item_extra[@]}")
      if [ -n "${TRANSFER_COHORT_ITEM_COHORT_NAME[${item}]}" ]; then
        # Each protein sub-item needs its own --cohort-name: two non-EC protein
        # items produce byte-identical cohorts, hence identical digests, and
        # would collide on the same output filename under the shared default.
        CMD+=(--cohort-name "${TRANSFER_COHORT_ITEM_COHORT_NAME[${item}]}")
      fi
      CMD+=(--out "${out_dir}")
      ;;
    pathway_budget)
      CMD+=("${TRANSFER_SCRIPTS}/02_pathway_budget.py"
            --arms "${item}" --device "cuda:${gpu}"
            --output-root "${out_dir}")
      ;;
    estimand_power)
      CMD+=("${TRANSFER_SCRIPTS}/03_estimand_power.py" measure
            --arms "${item}" --device "cuda:${gpu}"
            --output-root "${out_dir}")
      ;;
    circuit_primitives)
      CMD+=("${TRANSFER_SCRIPTS}/04_circuit_primitives.py"
            --arms "${CIRCUIT_ARMS[@]}" --device "cuda:${gpu}"
            --output-dir "${out_dir}")
      ;;
    relational_channel)
      CMD+=("${TRANSFER_SCRIPTS}/05_relational_channel.py"
            --arm "${item}" --device "cuda:${gpu}"
            --out "${out_dir}")
      ;;
    explanation_channel)
      # No --arm/--arms/--device/--dtype: CPU-only, whole panel. This branch
      # was missing until a re-read of this file caught it: run_explanation_
      # channel calls run_item_atomic (which always calls build_command) for
      # a stage build_command had no case for, so every invocation of stage
      # 06 would have hit the "unknown stage" fallback and failed outright.
      CMD+=("${TRANSFER_SCRIPTS}/06_explanation_channel.py" --out "${out_dir}")
      ;;
    convergence_control)
      CMD+=("${TRANSFER_SCRIPTS}/07_convergence_control.py"
            --device "cuda:${gpu}" --output-dir "${out_dir}"
            --backup-dir "${LOCAL_SCRATCH_ROOT}/${RUN_ID}/convergence_control_backup")
      ;;
    lens_family)
      CMD+=("${TRANSFER_SCRIPTS}/08_lens_family.py"
            --arms "${item}" --device "cuda:${gpu}"
            --output-root "${out_dir}")
      ;;
    probe_and_erasure)
      CMD+=("${TRANSFER_SCRIPTS}/09_probe_and_erasure.py"
            --arm "${item}" --device "cuda:${gpu}"
            --out "${out_dir}")
      ;;
    homology_control)
      # Protein-only by construction: the control asks whether a protein arm's
      # induction score is explained by training-set homology, which has no text
      # counterpart here. The arm list is 10_homology_control.py's OWN --arms
      # default, mirrored in the panel contract and checked against that file's
      # source by tests/test_transfer_stage_contract.py. This worker used to pass
      # its own protein list instead, which held four arms against the script's
      # three, so a campaign run and a direct run measured different panels while
      # the comment here claimed the opposite.
      CMD+=("${TRANSFER_SCRIPTS}/10_homology_control.py"
            --arms "${HOMOLOGY_ARMS[@]}" --device "cuda:${gpu}"
            --output-dir "${out_dir}")
      ;;
    induction_path_patching)
      CMD+=("${TRANSFER_SCRIPTS}/11_induction_path_patching.py"
            --arms "${PATH_PATCHING_ARMS[@]}" --device "cuda:${gpu}"
            --output-dir "${out_dir}")
      ;;
    paa_census)
      # Per arm on --census-arm; see the "14" entry in the header comment for
      # why --stages, --width and --text-arm are fixed here and every scale knob
      # is not. --width comes from the panel contract because the contract's
      # eligible arm list is declared against it: ProtGPT2 admits no full-width
      # cohort row at the entry point's own default of 512, so a width written
      # here and a width admitted there could disagree without either looking
      # wrong until a checkpoint was already on the GPU.
      #
      # --out gets a PER-ARM subdirectory, and this is not cosmetic. Unlike every
      # other per-arm stage, 14_paa_census.py names its principal artefacts after
      # the stage rather than after the arm -- census.json, causal.json,
      # selected_heads.json, census_matrices.npz, causal_matrices.npz,
      # paa_gate_report.json -- and its own census() docstring states the
      # invariant: "Arms are run in separate --out directories, so the per-arm
      # artefacts do not collide and need no renaming". Into one shared stage
      # directory, each arm would overwrite the previous arm's census, and each
      # item's resume manifest would then checksum a file some other arm wrote:
      # a silent cross-arm overwrite that verifies cleanly. run_item_atomic moves
      # produced files by their path relative to the temp directory and mkdir -p's
      # each parent, so the subdirectory survives into the results root. The path
      # does not reach provenance -- canonicalize_command normalizes --out out.
      CMD+=("${TRANSFER_SCRIPTS}/14_paa_census.py"
            --stages census causal
            --census-arm "${item}" --text-arm "${TEXT_ARM}"
            --width "${TRANSFER_PAA_CENSUS_WIDTH}"
            --device "cuda:${gpu}"
            --out "${out_dir}/${item}")
      ;;
    *)
      echo "build_command: unknown stage ${stage}" >&2
      exit 2
      ;;
  esac
  # Optional scale-parameter passthrough (--n-seq, --pool-size, --seeds and so
  # on -- each script names these differently, so this is a deliberately generic
  # append rather than enumerating every flag).
  #
  # Stage-wide args apply to every item of a stage; item-scoped args apply to
  # one. Both exist because a stage-wide-only knob cannot express "give the
  # residue cohort_power item a different --n-seq", and an operator who tried
  # would have moved all four items at once.
  local -a extra=()
  if [ -n "${STAGE_EXTRA_ARGS[${stage}]:-}" ]; then
    local -a stage_extra=()
    read -r -a stage_extra <<< "${STAGE_EXTRA_ARGS[${stage}]}"
    extra+=("${stage_extra[@]}")
  fi
  if [ -n "${ITEM_EXTRA_ARGS[${stage}/${item}]:-}" ]; then
    local -a item_specific=()
    read -r -a item_specific <<< "${ITEM_EXTRA_ARGS[${stage}/${item}]}"
    extra+=("${item_specific[@]}")
  fi
  if [ "${#extra[@]}" -gt 0 ]; then
    CMD+=("${extra[@]}")
    assert_no_duplicate_options "${stage}" "${item}" "${CMD[@]}"
  fi
}

# Refuses a passthrough flag that repeats one this worker already set for the
# same item.
#
# argparse takes the last occurrence, so a repeat is silently accepted and
# silently wins. For --n-seq that is merely confusing; for the flags this worker
# sets it is a measurement change with no record: --cohort-name decides the
# output filename, so overriding it on one item collides two items' cohorts on
# one path, and --dtype float32 and --skip-truncation each encode a measured
# reason (see build_command's cohort_power case). Overriding one of those is a
# decision to change what is measured, and it belongs in this file beside its
# reason rather than in an environment variable.
assert_no_duplicate_options() {
  local stage="$1" item="$2"
  shift 2
  local arg option seen=" "
  for arg in "$@"; do
    case "${arg}" in
      --*)
        option="${arg%%=*}"
        case "${seen}" in
          *" ${option} "*)
            echo "stage-args for ${stage}/${item} repeat ${option}, which this worker" >&2
            echo "already sets for this item. argparse would take the last one" >&2
            echo "silently. Change the flag in build_command, with its reason." >&2
            echo "  resolved command: $*" >&2
            exit 2
            ;;
        esac
        seen="${seen}${option} "
        ;;
    esac
  done
}

# Builds every command this run will issue, before the import preflight and long
# before any GPU is scheduled, so that a stage-args collision or an unknown item
# fails at argument-validation time rather than four stages into a campaign.
verify_commands_buildable() {
  local stage item
  local -a CMD
  log "command preflight: building every scheduled command"
  # The item namespace comes from the stage's DECLARED SCOPE, not from a branch
  # per stage. It used to be one `case` arm per per-arm stage -- five identical
  # bodies differing only in the stage name -- beneath a `*` fallback that built
  # every unlisted stage as `panel`. A per-arm stage added to the contract and
  # forgotten here was therefore not a build failure but a silently WRONG build:
  # the preflight would have checked `--census-arm panel` and passed. cohort_power
  # is the one stage whose items are not its arms (see TRANSFER_COHORT_ITEMS).
  for stage in "${REQUESTED_STAGES[@]}"; do
    if [ "${stage}" = cohort_power ]; then
      for item in "${COHORT_ITEMS[@]}"; do build_command "${stage}" "${item}" 0 "<pending>"; done
      continue
    fi
    case "${TRANSFER_STAGE_SCOPE[${stage}]}" in
      armless) build_command "${stage}" panel 0 "<pending>" ;;
      per_arm|control_anchored)
        for item in ${STAGE_ARMS_FOR[${stage}]:-}; do
          build_command "${stage}" "${item}" 0 "<pending>"
        done
        ;;
      *)
        # Panel-wide. One whose arm list is empty is skipped at dispatch
        # (run_panel_stage), so it is not built here.
        [ -n "${STAGE_ARMS_FOR[${stage}]:-}" ] \
          && build_command "${stage}" panel 0 "<pending>"
        ;;
    esac
  done
  log "command preflight passed"
}

# A requested stage produced no measurement. Recorded rather than logged-and-
# forgotten, because "measured nothing" and "measured successfully" are different
# facts and every route that conflated them reported success. See
# UNMEASURED_STAGES for the two routes.
record_unmeasured_stage() {
  local stage="$1" reason="$2"
  log "UNMEASURED stage=${stage} (${reason})"
  UNMEASURED_STAGES+=("${stage} (${reason})")
}

# Runs one requested stage and records that it was dispatched; logs and skips a
# stage this run did not ask for.
#
# Every tier-chain entry goes through this, which is what makes
# reconcile_dispatched_stages meaningful: a stage that reaches a runner by some
# other path would not be marked, and a stage with no path at all is caught.
dispatch_stage() {
  local stage="$1"
  shift
  if ! stage_requested "${stage}"; then
    log "stage ${stage} not in STAGES=${STAGES}; skipping"
    return 0
  fi
  DISPATCHED_STAGES+=("${stage}")
  "$@"
}

# Every requested stage must have been dispatched. The tier chain is eleven
# hand-written branches and nothing checked it against the contract-derived
# REQUESTED_STAGES, so a stage added to panel_contract.py and forgotten below
# passed the panel-contract verify, the command preflight and both import
# preflights -- all of which derive from the contract -- then never ran, and the
# campaign exited 0 with no line anywhere saying so.
#
# Only meaningful on the normal path: after an early exit the later stages
# genuinely have not run and the non-zero status already says so, which is why
# this is called from main and not from the EXIT trap.
reconcile_dispatched_stages() {
  local stage
  for stage in "${REQUESTED_STAGES[@]}"; do
    case " ${DISPATCHED_STAGES[*]:-} " in
      *" ${stage} "*) ;;
      *)
        record_unmeasured_stage "${stage}" \
          "requested, and accepted by every preflight, but no branch of this worker's tier chain dispatches it"
        ;;
    esac
  done
}

# Prints everything the campaign accumulated. Returns 1 if any of it means the
# campaign did not measure what it was asked to, 0 otherwise.
print_campaign_ledger() {
  local failed=0 failure skipped unmeasured
  if [ "${#DEFERRED_FAILURES[@]}" -gt 0 ]; then
    failed=1
    echo "campaign FAILED: run_id=${RUN_ID}; ${#DEFERRED_FAILURES[@]} deferred failure(s):" >&2
    for failure in "${DEFERRED_FAILURES[@]}"; do
      echo "  FAIL: ${failure}" >&2
    done
  fi
  if [ "${#SKIPPED_FOR_DATA[@]}" -gt 0 ]; then
    failed=1
    echo "campaign INCOMPLETE: run_id=${RUN_ID}; ${#SKIPPED_FOR_DATA[@]} item(s) never ran:" >&2
    for skipped in "${SKIPPED_FOR_DATA[@]}"; do
      echo "  SKIP-DATA: ${skipped}" >&2
    done
  fi
  if [ "${#UNMEASURED_STAGES[@]}" -gt 0 ]; then
    failed=1
    echo "campaign INCOMPLETE: run_id=${RUN_ID}; ${#UNMEASURED_STAGES[@]} requested stage(s) measured nothing:" >&2
    for unmeasured in "${UNMEASURED_STAGES[@]}"; do
      echo "  UNMEASURED: ${unmeasured}" >&2
    done
  fi
  return "${failed}"
}

finish_campaign() {
  CAMPAIGN_LEDGER_PRINTED=1
  if ! print_campaign_ledger; then
    echo "Completed items are retained. Re-run the same command after fixing the reported failures." >&2
    exit 1
  fi
  log "campaign complete: run_id=${RUN_ID} results=${RESULTS_ROOT} logs=${LOGS_ROOT}"
}

# finish_campaign is the only printer of the ledger and is reachable only by
# falling off the end of this file, but four call sites leave before then:
# run_stage_wave exits 1 on a hard item failure, verify_gpu_idle exits 2,
# build_command exits 2, assert_no_duplicate_options exits 2. Three items
# SKIP-DATA in tier 1 followed by a tier-3 item failure therefore discarded the
# SKIP-DATA record entirely -- the record that exists precisely so a data skip
# cannot be lost -- and left the operator with one failure message and no
# statement that three other items had never run at all.
#
# The trap only prints; it never calls `exit`, so bash exits with the status that
# triggered it and every exit code in this file is preserved exactly. It does not
# fire in the `&` subshells run_stage_wave forks, nor in command substitution,
# because bash runs an inherited EXIT trap only when the shell that installed it
# exits.
report_early_exit() {
  local status=$?
  if [ "${CAMPAIGN_LEDGER_PRINTED}" -eq 1 ]; then
    emit_exit_sentinel "${status}"
    return 0
  fi
  CAMPAIGN_LEDGER_PRINTED=1
  if print_campaign_ledger; then
    # Nothing had been accumulated -- a preflight refusal, say. Whatever exited
    # already said why, so adding a second voice here would only obscure it.
    emit_exit_sentinel "${status}"
    return 0
  fi
  echo "The above was accumulated before the worker exited early with status ${status}" >&2
  echo "(run_id=${RUN_ID:-unknown}). Completed items are retained; re-run the same" >&2
  echo "command after fixing the reported failure." >&2
  emit_exit_sentinel "${status}"
  return 0
}

#: Prefix of the line the controller reads this worker's real exit status from.
#:
#: **The access layer does not propagate it.** `h200_pod_exec.sh -- bash -c "exit
#: 7"` returns 0, measured directly on this deployment, so every non-zero exit of
#: this script has been invisible to `run_transfer_h200.sh`: a preflight refusal
#: that scheduled no GPU came back as "campaign complete" and status 0. That is
#: the false-success shape the whole ledger above exists to prevent, sitting one
#: layer above the ledger and defeating it.
#:
#: Fixed here rather than in the access layer because the access layer is outside
#: this repository and shared with other projects: a campaign's success must be
#: decidable from the campaign's own output. Emitted from the EXIT trap so it is
#: the last line whatever path exits, and its *absence* is itself a failure the
#: controller reports -- a worker killed mid-run, or a pod exec that never
#: started, prints no sentinel.
WORKER_EXIT_SENTINEL="TRANSFER_WORKER_EXIT="

emit_exit_sentinel() {
  printf '%s%s\n' "${WORKER_EXIT_SENTINEL}" "${1:-0}"
}
trap report_early_exit EXIT

# Re-derives panel_contract.sh from the live src/transfer/arms.py and refuses if
# the sourced copy disagrees. This is what makes the generated file a cache of
# the declaration rather than a second declaration: a panel edit that was not
# followed by `panel_contract.py --emit` stops the campaign here, before a GPU is
# scheduled, instead of running a stale arm list.
verify_panel_contract() {
  log "panel contract: re-deriving ${WORKER_DIR}/panel_contract.sh from src/transfer/arms.py"
  if ! "${TRANSFER_PYTHON}" "${TRANSFER_SCRIPTS}/panel_contract.py" --verify \
      --path "${WORKER_DIR}/panel_contract.sh"; then
    echo "panel_contract.sh does not match src/transfer/arms.py in this snapshot." >&2
    echo "Run: python scripts/transfer/panel_contract.py --emit, then re-freeze." >&2
    echo "No GPU has been scheduled." >&2
    exit 2
  fi
}

# Runs one (stage, item) pair to completion: skip-if-verified-complete,
# scoped data-path check, atomic temp-dir-then-move, manifest-then-rename.
# gpu="" means no GPU is needed (06_explanation_channel.py is CPU-only).
run_item_atomic() {
  local stage="$1" item="$2" gpu="$3"
  local final_dir tmp_dir log_file status=0 provenance
  local -a CMD produced=()

  final_dir="$(stage_final_dir "${stage}")"
  log_file="${LOGS_ROOT}/${stage}__${item}.log"

  # Compute this run's provenance before deciding whether to skip. "0" and
  # "<pending>" are placeholders for the GPU index and output path:
  # canonicalize_command normalizes both out of the comparison, so their
  # actual values here do not matter.
  build_command "${stage}" "${item}" "${gpu:-0}" "<pending>"
  provenance="$(provenance_record "${CMD[@]}")"

  if [ "${FORCE}" != "1" ] && item_is_complete "${stage}" "${item}" "${provenance}"; then
    log "skip  stage=${stage} item=${item} (matching code hash and parameters, checksums verified -- ${provenance})"
    return 0
  fi
  if [ -f "${final_dir}/.manifests/${item}.sha256" ]; then
    log "redo  stage=${stage} item=${item} ($(explain_incomplete "${stage}" "${item}" "${provenance}"))"
  fi

  # A missing input is a scheduling fact (this item cannot run until
  # staging catches up), not a computation defect -- it is logged and
  # skipped without writing a manifest, so it neither aborts the rest of
  # this campaign nor blocks a later rerun from picking it up once the
  # input lands. A genuine error from the measurement itself (the "${CMD[@]}"
  # invocation below) remains a hard failure, unchanged.
  if ! verify_item_data_paths "${stage}" "${item}"; then
    log "SKIP-DATA  stage=${stage} item=${item} (${MISSING_DATA_REASON})"
    return "${SKIP_DATA_STATUS}"
  fi

  mkdir -p "${final_dir}/.manifests"
  tmp_dir="$(mktemp -d "${final_dir}/.tmp.${stage}.${item}.XXXXXX")"

  if [ -n "${gpu}" ]; then
    verify_gpu_idle "${gpu}"
  fi
  build_command "${stage}" "${item}" "${gpu}" "${tmp_dir}"
  log "start stage=${stage} item=${item} gpu=${gpu:-none} log=${log_file}"
  "${CMD[@]}" >"${log_file}" 2>&1 || status=$?
  if [ "${status}" -ne 0 ]; then
    log "FAIL  stage=${stage} item=${item} gpu=${gpu:-none} status=${status} see ${log_file}"
    rm -rf -- "${tmp_dir}"
    return 1
  fi

  while IFS= read -r -d '' f; do
    produced+=("${f#"${tmp_dir}"/}")
  done < <(find "${tmp_dir}" -type f -print0)
  if [ "${#produced[@]}" -eq 0 ]; then
    log "FAIL  stage=${stage} item=${item} produced no output files under ${tmp_dir}"
    rm -rf -- "${tmp_dir}"
    return 1
  fi

  local rel
  for rel in "${produced[@]}"; do
    mkdir -p "${final_dir}/$(dirname "${rel}")"
    mv -f -- "${tmp_dir}/${rel}" "${final_dir}/${rel}"
  done
  ( cd "${final_dir}" && sha256sum -- "${produced[@]}" ) \
    > "${final_dir}/.manifests/.tmp.${item}.$$"
  mv -f -- "${final_dir}/.manifests/.tmp.${item}.$$" "${final_dir}/.manifests/${item}.sha256"
  printf '%s\n' "${provenance}" > "${final_dir}/.manifests/.tmp.${item}.provenance.$$"
  mv -f -- "${final_dir}/.manifests/.tmp.${item}.provenance.$$" \
    "${final_dir}/.manifests/${item}.provenance"
  rmdir "${tmp_dir}" 2>/dev/null || rm -rf -- "${tmp_dir}"

  log "done  stage=${stage} item=${item} gpu=${gpu:-none} files=${#produced[@]}"
  return 0
}

# Runs one stage across a list of items, one item per GPU, in waves sized
# to len(GPU_LIST). Exits the whole worker non-zero the moment any item
# fails, so a later tier never starts on top of a partial earlier one.
run_stage_wave() {
  local stage="$1"
  shift
  local -a items=("$@")
  local -a pids=() pid_items=()
  local i j gpu item status fail=0 skipped=0 n_gpu="${#GPU_LIST[@]}"

  if [ "${#items[@]}" -eq 0 ]; then
    record_unmeasured_stage "${stage}" \
      "requested, but no item to run: no arm in ARMS=${ARMS} is eligible for this stage"
    return 0
  fi
  for i in "${!items[@]}"; do
    item="${items[$i]}"
    gpu="${GPU_LIST[$(( i % n_gpu ))]}"
    run_item_atomic "${stage}" "${item}" "${gpu}" &
    pids+=("$!")
    pid_items+=("${item}")
    if [ $(( (i + 1) % n_gpu )) -eq 0 ] || [ "${i}" -eq $(( ${#items[@]} - 1 )) ]; then
      for j in "${!pids[@]}"; do
        status=0
        wait "${pids[$j]}" || status=$?
        case "${status}" in
          0) ;;
          "${SKIP_DATA_STATUS}")
            SKIPPED_FOR_DATA+=("${stage}/${pid_items[$j]} (required data not staged; see SKIP-DATA log)")
            skipped=1
            ;;
          *)
            echo "stage ${stage} failed for item ${pid_items[$j]}" >&2
            fail=1
            ;;
        esac
      done
      pids=()
      pid_items=()
    fi
  done
  if [ "${fail}" -ne 0 ]; then
    echo "stage ${stage} had at least one failing item; stopping the campaign" >&2
    exit 1
  fi
  if [ "${skipped}" -ne 0 ]; then
    log "stage ${stage} incomplete: at least one item was skipped for missing data"
  else
    log "stage ${stage} complete for items: ${items[*]}"
  fi
}

run_item_serial() {
  local stage="$1" item="$2" gpu="$3" status=0
  run_item_atomic "${stage}" "${item}" "${gpu}" || status=$?
  if [ "${status}" -eq "${SKIP_DATA_STATUS}" ]; then
    SKIPPED_FOR_DATA+=("${stage}/${item} (required data not staged; see SKIP-DATA log)")
    return 0
  fi
  return "${status}"
}

# 03 measure: text arm alone first, then the protein arms (evidence
# discipline rule 1), then the CPU-only recommend aggregation. recommend
# only reads already-written per-arm JSON, so it needs no data-path check.
run_estimand_power() {
  local -a arms=("$@")
  local -a protein_arms=()
  local arm out_dir log_file status=0 provenance

  out_dir="$(stage_final_dir estimand_power)"
  if [ "${#arms[@]}" -eq 0 ]; then
    record_unmeasured_stage estimand_power \
      "requested, but no arm in ARMS=${ARMS} is eligible for this stage"
    return 0
  fi
  # The text control is not optional here: `recommend` anchors the panel verdict
  # on it, and `measure` is ordered so that it runs first (evidence discipline
  # rule 1). Refusing is better than silently anchoring on whatever else is
  # present, which is how a control-anchored aggregation received a whole panel.
  case " ${arms[*]} " in
    *" ${TEXT_ARM} "*) ;;
    *)
      echo "estimand_power: the text control ${TEXT_ARM} is not among the eligible" >&2
      echo "arms (${arms[*]}), so no panel verdict can be anchored" >&2
      exit 2
      ;;
  esac
  log "stage estimand_power: measuring the text arm ${TEXT_ARM} first, per evidence discipline rule 1"
  run_stage_wave estimand_power "${TEXT_ARM}"
  for arm in "${arms[@]}"; do
    [ "${arm}" = "${TEXT_ARM}" ] || protein_arms+=("${arm}")
  done
  if [ "${#protein_arms[@]}" -gt 0 ]; then
    run_stage_wave estimand_power "${protein_arms[@]}"
  fi

  # `recommend` is an attainability aggregation, not a survey: evidence
  # discipline rule 1 says a gate is applied to a protein arm only after it has
  # been shown attainable on *the* text control, and 03_estimand_power.py's
  # recommend() enforces that literally -- it raises unless exactly one arm in
  # its --arms list is text. Handing it the whole panel gave
  # "the attainability check needs exactly one text positive control, found
  # ['llama-3.2-3b', 'gpt2-xl', 'gpt2-large', 'qwen2.5-0.5b', 'gpt2-medium',
  # 'gpt2', 'dialogpt-small']" and lost the run. The extra text arms are still
  # measured -- `measure` ran for every arm and each has its own per-estimand
  # `powered` flag on disk -- so nothing is dropped from the record; what is
  # scoped here is which single arm the panel verdict is anchored on, which is
  # TEXT_ARM by definition.
  local -a recommend_arms=("${TEXT_ARM}")
  for arm in "${arms[@]}"; do
    if [ "${arm}" != "${TEXT_ARM}" ] && [ "$(arm_modality "${arm}")" = protein ]; then
      recommend_arms+=("${arm}")
    fi
  done
  log "stage estimand_power: recommend anchored on the text control ${TEXT_ARM} over ${recommend_arms[*]}"

  # recommend takes no --device/--dtype; its own arguments are just the arm list
  # plus paths, and canonicalize_command normalizes the paths out.
  #
  # Its provenance must also cover its INPUTS, which no other item needs. Every
  # other item's output is a function of its own command; recommend's is a
  # function of the per-arm measure outputs already on disk. Keyed on the command
  # alone, changing ARGS_ESTIMAND_POWER re-ran `measure` for every arm and then
  # SKIPPED recommend as complete, leaving a recommendation.json derived from
  # measure outputs that no longer existed -- a stale panel verdict that verifies
  # cleanly against its own checksum. The consumed manifests are folded in, so a
  # measure re-run invalidates the aggregation that reads it.
  local -a input_manifests=() absent_measures=()
  for arm in "${recommend_arms[@]}"; do
    if [ ! -f "${out_dir}/.manifests/${arm}.sha256" ]; then
      absent_measures+=("${arm}")
    fi
    input_manifests+=("${out_dir}/.manifests/${arm}.sha256")
  done
  # A missing per-arm manifest is exactly what a legitimate SKIP-DATA above
  # produces, and it used to end the worker without a word: `cat` on an absent
  # file exits 1, `2>/dev/null` hid the message, `pipefail` made the whole
  # pipeline non-zero and `set -e` exited the process here -- before tiers 3 and
  # 4, and before finish_campaign could print the SKIP-DATA summary that
  # explained why. Refuse explicitly instead. `recommend` cannot anchor a panel
  # verdict on a measure that never ran, so this is a refusal, not a crash and
  # not an aggregation over whatever happens to be on disk.
  if [ "${#absent_measures[@]}" -gt 0 ]; then
    log "SKIP-DATA  stage=estimand_power item=recommend (no measure output for: ${absent_measures[*]})"
    SKIPPED_FOR_DATA+=(
      "estimand_power/recommend (cannot anchor a panel verdict: no measure output for ${absent_measures[*]})"
    )
    return 0
  fi
  local inputs_digest
  if ! inputs_digest="$(cat -- "${input_manifests[@]}" | sha256sum | awk '{print $1}')"; then
    echo "estimand_power: could not read the measure manifests recommend aggregates" >&2
    echo "  ${input_manifests[*]}" >&2
    exit 2
  fi
  provenance="$(provenance_record recommend --arms "${recommend_arms[@]}" \
    --measure-inputs-sha256 "${inputs_digest}" \
    --results-root "<pending>" --output "<pending>")"
  if [ "${FORCE}" != "1" ] && item_is_complete estimand_power recommend "${provenance}"; then
    log "skip  stage=estimand_power item=recommend (matching code hash and parameters, checksums verified -- ${provenance})"
    return 0
  fi
  if [ -f "${out_dir}/.manifests/recommend.sha256" ]; then
    log "redo  stage=estimand_power item=recommend ($(explain_incomplete estimand_power recommend "${provenance}"))"
  fi
  mkdir -p "${out_dir}/.manifests"
  local tmp_dir
  tmp_dir="$(mktemp -d "${out_dir}/.tmp.estimand_power.recommend.XXXXXX")"
  log_file="${LOGS_ROOT}/estimand_power__recommend.log"
  log "start stage=estimand_power item=recommend (CPU-only aggregation, no GPU) log=${log_file}"
  "${TRANSFER_PYTHON}" "${TRANSFER_SCRIPTS}/03_estimand_power.py" recommend \
      --arms "${recommend_arms[@]}" \
      --results-root "${out_dir}" \
      --output "${tmp_dir}/recommendation.json" \
      >"${log_file}" 2>&1 || status=$?
  if [ "${status}" -ne 0 ]; then
    # Deferred, not fatal. recommend is a CPU-only aggregation over per-arm
    # measure outputs that are already written, verified and manifested; it
    # produces no input any later stage reads. Exiting here cost tier 3 -- six
    # GPU stages -- to a failure in a CPU aggregation at the end of tier 2, and
    # re-running the campaign to recover them re-did nothing but re-verify
    # manifests, because every completed item skips. The failure is recorded and
    # the worker exits non-zero at the end, so the campaign is still reported as
    # failed and no manifest is written for this item.
    log "FAIL  stage=estimand_power item=recommend status=${status} see ${log_file}"
    log "      deferred: recommend feeds no later stage, so tier 3 continues; the worker will exit non-zero"
    rm -rf -- "${tmp_dir}"
    DEFERRED_FAILURES+=("estimand_power/recommend (status ${status}, see ${log_file})")
    return 0
  fi
  mv -f -- "${tmp_dir}/recommendation.json" "${out_dir}/recommendation.json"
  ( cd "${out_dir}" && sha256sum -- recommendation.json ) \
    > "${out_dir}/.manifests/.tmp.recommend.$$"
  mv -f -- "${out_dir}/.manifests/.tmp.recommend.$$" "${out_dir}/.manifests/recommend.sha256"
  printf '%s\n' "${provenance}" > "${out_dir}/.manifests/.tmp.recommend.provenance.$$"
  mv -f -- "${out_dir}/.manifests/.tmp.recommend.provenance.$$" \
    "${out_dir}/.manifests/recommend.provenance"
  rmdir "${tmp_dir}" 2>/dev/null || rm -rf -- "${tmp_dir}"
  log "done  stage=estimand_power item=recommend"
}

# 06: no --arm/--arms, no --device, CPU-only, single job for the panel.
run_explanation_channel() {
  run_item_serial explanation_channel panel ""
}

# A panel-wide stage with no eligible arm must not be dispatched: its `--arms`
# would expand to nothing and the entry point would fall back to its own default,
# which is a different panel from the one this campaign resolved. Skipping with a
# reason is the only correct answer.
run_panel_stage() {
  local stage="$1" gpu="$2"
  shift 2
  if [ "$#" -eq 0 ]; then
    record_unmeasured_stage "${stage}" \
      "requested, but no arm in ARMS=${ARMS} is eligible; not dispatched, because an empty --arms would fall back to the entry point's own default panel"
    return 0
  fi
  run_item_serial "${stage}" panel "${gpu}"
}

# 04 and 07 also run once, covering the whole arm list (04) or the whole
# ladder (07) inside one process; see build_command and the header comment
# for why per-arm/per-member dispatch is unsafe for these two.
run_circuit_primitives() {
  run_panel_stage circuit_primitives "${GPU_LIST[0]}" "${CIRCUIT_ARMS[@]}"
}

run_convergence_control() {
  run_item_serial convergence_control panel "${GPU_LIST[0]}"
}

# 10 and 11 are panel-scoped like circuit_primitives: each builds its own
# cohort once and sweeps arms internally, so splitting them across GPUs would
# rebuild the cohort per shard and break the shared-probe design both rely on.
run_homology_control() {
  run_panel_stage homology_control "${GPU_LIST[0]}" "${HOMOLOGY_ARMS[@]}"
}

run_induction_path_patching() {
  run_panel_stage induction_path_patching "${GPU_LIST[0]}" "${PATH_PATCHING_ARMS[@]}"
}

# ----------------------------------------------------------------- main

KNOWN_STAGES="${TRANSFER_STAGE_ORDER}"
IFS=',' read -r -a REQUESTED_STAGES <<< "${STAGES}"
if [ "${#REQUESTED_STAGES[@]}" -eq 0 ]; then
  echo "STAGES must not be empty" >&2
  exit 2
fi
for stage in "${REQUESTED_STAGES[@]}"; do
  case " ${KNOWN_STAGES} " in
    *" ${stage} "*) ;;
    *) echo "unknown stage: ${stage} (known stages: ${KNOWN_STAGES})" >&2; exit 2 ;;
  esac
done
reject_duplicate_values STAGES "${REQUESTED_STAGES[@]}"

# Whether a given stage is part of this run, for gating the tier-execution
# calls in main below.
stage_requested() {
  case " ${REQUESTED_STAGES[*]} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

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
if stage_requested estimand_power; then
  case " ${ARM_LIST[*]} " in
    *" ${TEXT_ARM} "*) ;;
    *)
      echo "ARMS must include the text arm ${TEXT_ARM}: 03_estimand_power.py" >&2
      echo "requires it to run first (evidence discipline rule 1)" >&2
      exit 2
      ;;
  esac
fi
if ! [[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPUS must be a comma-separated list of integers, got: ${GPUS}" >&2
  exit 2
fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
reject_duplicate_values GPUS "${GPU_LIST[@]}"

TEXT_ARMS=()
PROTEIN_ARMS=()
for arm in "${ARM_LIST[@]}"; do
  if [ "$(arm_modality "${arm}")" = text ]; then
    TEXT_ARMS+=("${arm}")
  else
    PROTEIN_ARMS+=("${arm}")
  fi
done

# Per-stage arm lists, every one of them from the panel contract's arm_can_run
# predicate rather than from a `case` on the arm's name. Refusals are logged as
# they are computed, so an operator reading the log sees the panel each stage
# actually ran and why it differs from ARMS, instead of having to reconstruct it.
#
# Two of these lists were previously wrong. lens_family's hand-written exclusion
# happened to match src.transfer.scaling.LENS_ARCHITECTURES; relational_channel's
# hand-written inclusion did not -- it named zymctrl and progen2-medium, omitting
# progen2-base, which is protein, residue-tokenised and carries the `relational`
# capability, so a campaign measured a three-arm relational panel as a two-arm
# one. homology_control's list was wrong in the other direction: the worker
# passed its own four-arm protein list to a script whose own --arms default names
# three, so a campaign run and a direct run measured different panels.
declare -A STAGE_ARMS_FOR=()
for stage in ${TRANSFER_STAGE_ORDER}; do
  case "${TRANSFER_STAGE_SCOPE[${stage}]}" in
    armless) continue ;;
  esac
  stage_eligible_arms "${stage}" "${ARM_LIST[@]}"
  STAGE_ARMS_FOR["${stage}"]="${STAGE_ARMS_OUT[*]:-}"
done
read -r -a LENS_ARMS <<< "${STAGE_ARMS_FOR[lens_family]:-}"
read -r -a RELATIONAL_ARMS <<< "${STAGE_ARMS_FOR[relational_channel]:-}"
read -r -a PATHWAY_ARMS <<< "${STAGE_ARMS_FOR[pathway_budget]:-}"
read -r -a ESTIMAND_ARMS <<< "${STAGE_ARMS_FOR[estimand_power]:-}"
read -r -a PROBE_ARMS <<< "${STAGE_ARMS_FOR[probe_and_erasure]:-}"
read -r -a CIRCUIT_ARMS <<< "${STAGE_ARMS_FOR[circuit_primitives]:-}"
read -r -a HOMOLOGY_ARMS <<< "${STAGE_ARMS_FOR[homology_control]:-}"
read -r -a PATH_PATCHING_ARMS <<< "${STAGE_ARMS_FOR[induction_path_patching]:-}"
read -r -a PAA_CENSUS_ARMS <<< "${STAGE_ARMS_FOR[paa_census]:-}"

# cohort_power's four-way split, also from the contract: by vocabulary regime for
# the truncation-curve guard, and the residue arms isolated further for their
# float32 override. Each item's arm list, extra flags and --cohort-name are
# declared in panel_contract.py::cohort_power_items with the measured reason.
declare -A COHORT_ITEM_ARMS_FOR=()
COHORT_ITEMS=()
read -r -a COHORT_ELIGIBLE <<< "${STAGE_ARMS_FOR[cohort_power]:-}"
for item in ${TRANSFER_COHORT_ITEMS}; do
  item_arms=()
  for arm in ${TRANSFER_COHORT_ITEM_ARMS[${item}]}; do
    case " ${COHORT_ELIGIBLE[*]} " in
      *" ${arm} "*) item_arms+=("${arm}") ;;
    esac
  done
  if [ "${#item_arms[@]}" -gt 0 ]; then
    COHORT_ITEMS+=("${item}")
    COHORT_ITEM_ARMS_FOR["${item}"]="${item_arms[*]}"
  fi
done
unset item item_arms arm

log "InterpretabilityTransfer H200 worker"
log "run_id:             ${RUN_ID}"
log "snapshot_dir:        ${SNAPSHOT_DIR}"
log "results_root:        ${RESULTS_ROOT}"
log "logs_root:           ${LOGS_ROOT}"
log "arms:                ${ARM_LIST[*]}"
log "gpus:                ${GPU_LIST[*]}"
log "text_arm:            ${TEXT_ARM}"
log "stages:              ${REQUESTED_STAGES[*]}"
log "text_arms:           ${TEXT_ARMS[*]:-(none)}"
log "protein_arms:        ${PROTEIN_ARMS[*]:-(none)}"
log "cohort_power_items:  ${COHORT_ITEMS[*]:-(none)}"
log "panel_contract:      ${TRANSFER_CONTRACT_SCHEMA}"
for stage in ${TRANSFER_STAGE_ORDER}; do
  [ "${TRANSFER_STAGE_SCOPE[${stage}]}" = armless ] && continue
  stage_requested "${stage}" || continue
  log "  ${stage} arms:      ${STAGE_ARMS_FOR[${stage}]:-(none)}"
done
log "expected_gpu_count:  ${EXPECTED_GPU_COUNT:-(auto, from nvidia-smi)}"
log "min_free_mem_mib:    ${MIN_FREE_MEM_MIB} (soft warning only)"
log "transfer_python:     ${TRANSFER_PYTHON}"
log "force:               ${FORCE}"

mkdir -p "${LOGS_ROOT}"
verify_panel_contract
verify_commands_buildable
verify_entry_points_importable
verify_entry_points_parse_args
verify_gpfs_read_write
verify_gpus
mkdir -p "${RESULTS_ROOT}"
record_host_state "before the campaign"

# Every stage not in REQUESTED_STAGES is logged and left alone rather than
# dispatched -- this is how a campaign scoped to what is staged today (see
# scripts/transfer/README.md's "Controller Environment") avoids the stages
# that need data that has not landed yet. Re-running the same command once
# more data lands picks up exactly the stages/items that were skipped,
# without --force and without editing this file, because a skip -- for
# either reason, an unrequested stage or (see verify_item_data_paths) a
# missing input -- never writes a completion manifest.
#
# Every line below goes through dispatch_stage, which is what makes
# reconcile_dispatched_stages able to catch a contract stage this chain forgets.
log "tier 1: cohort_power (prerequisite; must pass before anything consumes the cohort)"
dispatch_stage cohort_power run_stage_wave cohort_power "${COHORT_ITEMS[@]}"

log "tier 2: pathway_budget, then estimand_power"
dispatch_stage pathway_budget run_stage_wave pathway_budget "${PATHWAY_ARMS[@]}"
dispatch_stage estimand_power run_estimand_power "${ESTIMAND_ARMS[@]}"

log "tier 3: circuit_primitives, relational_channel, explanation_channel, convergence_control, lens_family, probe_and_erasure"
dispatch_stage circuit_primitives run_circuit_primitives
dispatch_stage relational_channel run_stage_wave relational_channel "${RELATIONAL_ARMS[@]}"
dispatch_stage explanation_channel run_explanation_channel
dispatch_stage convergence_control run_convergence_control
dispatch_stage lens_family run_stage_wave lens_family "${LENS_ARMS[@]}"
dispatch_stage probe_and_erasure run_stage_wave probe_and_erasure "${PROBE_ARMS[@]}"

log "tier 4: homology_control, induction_path_patching, paa_census"
dispatch_stage homology_control run_homology_control
dispatch_stage induction_path_patching run_induction_path_patching
# Per arm, so it fills every GPU in the wave rather than holding one card the way
# the two panel-scoped stages above do. Last in the tier for that reason: it is
# the only stage here that can absorb the whole allocation.
dispatch_stage paa_census run_stage_wave paa_census "${PAA_CENSUS_ARMS[@]}"

record_host_state "after the campaign"

reconcile_dispatched_stages
finish_campaign
