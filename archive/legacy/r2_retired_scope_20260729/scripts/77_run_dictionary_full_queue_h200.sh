#!/usr/bin/env bash
set -euo pipefail

# Run one fixed GPU queue from the terminally eligible P0-2 full panel.

ROOT="/gpfs/jiaotongdamoxing/zhk_zip/biocc/npj_revision_20260717"
PROJECT="${ROOT}/code_p0_2_dictionary_controls_bf16_r3/r2_interpretability_transfer"
CODE_ARCHIVE="${ROOT}/code_archives/r2_p0_2_dictionary_controls_bf16_payload_r3.tar"
TREE_MANIFEST="${PROJECT}/CODE_CONTENT_SHA256SUMS"
RUNNER="${PROJECT}/scripts/58_run_dictionary_controls.py"
MODULE="${PROJECT}/src/revision/dictionary_controls.py"
PROFILE="${PROJECT}/configs/p0_2_dictionary_controls_production_profile.json"
CACHE_ROOT="${ROOT}/p0_2_exact_cache_bf16_r3"
SCREENING_ROOT="${ROOT}/p0_2_dictionary_controls_bf16_r3/screening"
OUTPUT_ROOT="${ROOT}/p0_2_dictionary_controls_bf16_r3/full"
CODE_ARCHIVE_SHA256="6f191b554de901c7be25968f2fc96d989ae7d5d0bcb2a1c9285fdd1c3b840e44"
TREE_MANIFEST_SHA256="486ba4e9c00a7a1a88a58a31691095e72aaf45f39745ad3117b88c694de579d4"
PROFILE_SHA256="eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2"
RUNNER_SHA256="56ca3c4d8e230ea6ef5cf36d394564f747fa2c5bcb4f9b837dd3e5825e6401b8"
MODULE_SHA256="347a095c2e18a429e09011f84a45bd40b1cf5d46ebf38398e6e2a7afe57c6596"
GPU_ID="${GPU_ID:?set GPU_ID to 0, 1, 2 or 3}"
QUEUE_ID="${QUEUE_ID:?set QUEUE_ID to the matching fixed queue}"
QUEUE_MODE="${QUEUE_MODE:?set QUEUE_MODE to fresh or resume}"
POD_NAME="damoxing-zhk-zipbio-master-0"
NODE_NAME="i-d5cvmv6heob1nidq4ujg"
GIT_COMMIT="d765d4bf761d566fb45b6318c6067c9981cf96ad"

export PYTHONDONTWRITEBYTECODE=1

if [[ ! "${GPU_ID}" =~ ^[0-3]$ ]] || [ "${QUEUE_ID}" != "${GPU_ID}" ]; then
  echo "GPU_ID and QUEUE_ID must be the same integer in 0..3" >&2
  exit 2
fi
if [ "${QUEUE_MODE}" != fresh ] && [ "${QUEUE_MODE}" != resume ]; then
  echo "QUEUE_MODE must be exactly fresh or resume" >&2
  exit 2
fi

require_sha() {
  local expected="$1"
  local path="$2"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

verify_deployment() {
  local archived_manifest_sha

  require_sha "${CODE_ARCHIVE_SHA256}" "${CODE_ARCHIVE}"
  require_sha "${TREE_MANIFEST_SHA256}" "${TREE_MANIFEST}"
  require_sha "${PROFILE_SHA256}" "${PROFILE}"
  require_sha "${RUNNER_SHA256}" "${RUNNER}"
  require_sha "${MODULE_SHA256}" "${MODULE}"
  archived_manifest_sha="$({
    tar -xOf "${CODE_ARCHIVE}" \
      r2_interpretability_transfer/CODE_CONTENT_SHA256SUMS
  } | sha256sum | awk '{print $1}')"
  test "${archived_manifest_sha}" = "${TREE_MANIFEST_SHA256}"
  if find "${PROJECT}" -type l -print -quit | grep -q .; then
    echo "deployed r3 tree contains a symbolic link" >&2
    exit 2
  fi
  (
    cd "${PROJECT}"
    sha256sum --check --strict CODE_CONTENT_SHA256SUMS >/dev/null
    manifest_inventory="$(mktemp)"
    deployed_inventory="$(mktemp)"
    trap 'rm -f "${manifest_inventory}" "${deployed_inventory}"' EXIT
    cut -c67- CODE_CONTENT_SHA256SUMS | LC_ALL=C sort >"${manifest_inventory}"
    find . -type f ! -name CODE_CONTENT_SHA256SUMS -print | LC_ALL=C sort \
      >"${deployed_inventory}"
    cmp -s "${manifest_inventory}" "${deployed_inventory}"
  )
}

cache_identity() {
  case "$1" in
    protgpt2)
      echo "d3fa68212612bb42ed9d75c0b49a606db44a9171c00b0662c05b41f48f63dd34 c5e323df9618c14db9ec4e19332dc1b162a4a026b68381b2b1fd3c7954a73225 ddf4ea1e99d5b382124b31b80fba1bcd99c4e84a9c48b00e28827c955c4f87dc 221184000000"
      ;;
    zymctrl)
      echo "26307ed3694c884543d3033cd99e6cee52cd5441c9ce7f37f6f9a4f1a48dbc70 0b4c4e9040556d73dee68421cff13cc1497e4c8348e03a4c72f9eb0bfd0fb013 cb91a518627e516dfe7cd66dc9fbd61afcb60526b21df392f811f7b9bc1050de 221184000000"
      ;;
    progen2-medium)
      echo "e51c9dacfc7cbb56fb8f860800a4e2315686838853e34f1905d95fa757d72ea9 31b34aef4a059b061b4979a5afa5a4b07ccc0278b37ecac3e2a69d7153171a80 fc3d0e80016a672d6e58a9f9c6f0f92cf5394c464ca4c633929c4b98880d3166 199065600000"
      ;;
    *) return 2 ;;
  esac
}

verify_cache_receipt() {
  local model="$1"
  local manifest_sha receipt_sha content_sha payload_bytes
  local receipt manifest

  read -r manifest_sha receipt_sha content_sha payload_bytes \
    < <(cache_identity "${model}")
  receipt="${CACHE_ROOT}/${model}/completion_receipt.json"
  manifest="${CACHE_ROOT}/${model}/manifest.json"
  require_sha "${receipt_sha}" "${receipt}"
  require_sha "${manifest_sha}" "${manifest}"
  python3 - "${receipt}" "${manifest}" "${model}" "${PROFILE_SHA256}" \
    "${content_sha}" "${payload_bytes}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, manifest_path = map(Path, sys.argv[1:3])
model, profile_sha, content_sha = sys.argv[3:6]
payload_bytes = int(sys.argv[6])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected = {
    "schema_version": "r2_dictionary_cache_completion_receipt_v3",
    "status": "verified_complete",
    "model_name": model,
    "profile_sha256": profile_sha,
    "activation_payload_bytes": payload_bytes,
    "cache_storage_dtype": "float16",
    "model_inference_dtype": "bfloat16",
    "observed_model_parameter_dtypes": ["bfloat16"],
    "model_inference_dtype_verified": True,
    "code_content_inventory_verified": True,
}
if any(receipt.get(key) != value for key, value in expected.items()):
    raise SystemExit("cache completion receipt violates the frozen contract")
if Path(receipt["cache_manifest_path"]) != manifest_path:
    raise SystemExit("cache receipt points to a different manifest")
if sha256(manifest_path) != receipt["cache_manifest_sha256"]:
    raise SystemExit("cache receipt manifest hash mismatch")
report_path = Path(receipt["execution_report_path"])
if not report_path.is_file() or sha256(report_path) != receipt["execution_report_sha256"]:
    raise SystemExit("cache receipt execution-report hash mismatch")
if manifest.get("content_sha256") != content_sha:
    raise SystemExit("cache manifest content identity mismatch")
if receipt.get("cache_content_sha256") != content_sha:
    raise SystemExit("cache receipt content identity mismatch")
PY
}

screening_identity() {
  case "$1/$2" in
    protgpt2/relu_l1_sae)
      echo "61e1837e94c41d72e5e8d3bd3bfeed9339c6ce6d67b6cf5ee0459436efe23bf7 2c03f22b90f4e7bdef57f7cd931df3e80fa06b4d13f49df5ab87eaeb0d5c036b sparsity_match_failure none"
      ;;
    protgpt2/gated_sae)
      echo "6c4f954bb21ad10a1369a2b102646ae7813d450e738c3dcf6edea10bd1140a28 ed5ad704f374d1cde3eddf4013bc692aff10aba84fd210ed143171cf4e6c4cfc sparsity_match_failure none"
      ;;
    zymctrl/relu_l1_sae)
      echo "96e19e97e62af904959973b85646aed595d23635375e47888ad1cff9acf26f1c f9c6f2c959de729bbf1d4e20e7ed73f2e5f32f2e579c7ddbd758b86291144e84 completed_validation_screening 0,1e-05,0.0,0.0,131.2233588888889,0.5899043000429988,46327294afbad0f464e20db8073870d66367ebeb94a1fceb669c017a6ddb7ea2"
      ;;
    zymctrl/gated_sae)
      echo "ea354362e10801192c4ff4c52a22a462a8d9432c859fd9c79a3c770f881ebe4e d772e7311a2374d8870949ab9df2da0887b56e2ac90b34de65ca352109c7514e completed_validation_screening 3,3e-05,1.0,0.0,120.02304111111111,0.5929956473037449,7b7de87a370aa49f68b2525a162707712675fade013e9c36efdb5aed74eef1b0"
      ;;
    progen2-medium/relu_l1_sae)
      echo "ceca32f71131f7109980db1475265c7f44c14330275a5dce781c4a990525b61f e89a15c30635da43a6cfa2c5ff4e1ad2418d692fc3cd0611d43f193ce6a8860f sparsity_match_failure none"
      ;;
    progen2-medium/gated_sae)
      echo "8d49d227021c32e65ce831c428649c6648df0dc9e5d2aaa03852f55f194dadd4 7acb5c200ba76eb3988573ce0972daca0fa680adf4c39cb71486218d13ddafeb completed_validation_screening 0,1e-05,0.1,0.0,132.16376074074074,0.4397525937186076,475283ed26edcd3fb2da4af948dfa78e34aa0bb0205f1f92ad480d433f8151ee"
      ;;
    *) return 2 ;;
  esac
}

verify_screening() {
  local model="$1"
  local method="$2"
  local result_sha manifest_sha status selection
  local cache_manifest_sha cache_receipt_sha cache_content_sha payload_bytes
  local directory result manifest candidate_index checkpoint_sha checkpoint

  read -r result_sha manifest_sha status selection \
    < <(screening_identity "${model}" "${method}")
  read -r cache_manifest_sha cache_receipt_sha cache_content_sha payload_bytes \
    < <(cache_identity "${model}")
  directory="${SCREENING_ROOT}/${model}/${method}"
  result="${directory}/results.json"
  manifest="${directory}/run_manifest.json"
  require_sha "${result_sha}" "${result}"
  require_sha "${manifest_sha}" "${manifest}"
  python3 - "${result}" "${manifest}" "${model}" "${method}" "${status}" \
    "${selection}" "${result_sha}" "${PROFILE_SHA256}" \
    "${cache_manifest_sha}" "${cache_content_sha}" "${RUNNER_SHA256}" \
    "${MODULE_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

(
    result_path,
    manifest_path,
    model,
    method,
    status,
    selection_spec,
    result_sha,
    profile_sha,
    cache_manifest_sha,
    cache_content_sha,
    runner_sha,
    module_sha,
) = sys.argv[1:]
result = json.loads(Path(result_path).read_text(encoding="utf-8"))
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
expected_result = {
    "schema_version": "r2_dictionary_control_results_v1",
    "status": status,
    "p0_2_eligible": False,
    "stage": "screening",
    "method": method,
    "model_name": model,
    "run_seed": 20260717,
    "profile_sha256": profile_sha,
    "cache_content_sha256": cache_content_sha,
    "test_evaluation_count": 0,
}
if any(result.get(key) != value for key, value in expected_result.items()):
    raise SystemExit(f"screening result contract mismatch: {model}/{method}")
candidate_count = {"relu_l1_sae": 5, "gated_sae": 10}[method]
candidate_validation = result.get("candidate_validation", [])
if len(candidate_validation) != candidate_count:
    raise SystemExit("screening result has an incomplete candidate grid")
if len(result.get("selection_rows", [])) != candidate_count * 5:
    raise SystemExit("screening result has an incomplete threshold grid")
if [row.get("candidate_index") for row in candidate_validation] != list(
    range(candidate_count)
):
    raise SystemExit("screening candidate identities are incomplete or reordered")
for row in candidate_validation:
    training = row.get("training") if isinstance(row, dict) else None
    if (
        row.get("stage") != "screening"
        or not isinstance(training, dict)
        or training.get("post_candidate_accelerator_memory_allocated_bytes")
        is None
        or training["post_candidate_accelerator_memory_allocated_bytes"]
        > 128 * 1024**2
    ):
        raise SystemExit("screening candidate violates the terminal lifecycle contract")
if status == "sparsity_match_failure":
    if selection_spec != "none":
        raise SystemExit("internal failure-selection pin is malformed")
    if result.get("reason") != "no validation candidate achieved L0 within [115.2, 140.8]":
        raise SystemExit("screening failure reason changed")
    if "selected_validation_configuration" in result:
        raise SystemExit("failed screening unexpectedly selected a configuration")
    if "selected_checkpoint" in result:
        raise SystemExit("failed screening unexpectedly selected a checkpoint")
    if any(
        115.2 <= row.get("validation_l0_mean", float("inf")) <= 140.8
        for row in result["selection_rows"]
    ):
        raise SystemExit("screening failure contains an eligible L0 row")
    checkpoint_sha = None
else:
    fields = selection_spec.split(",")
    if len(fields) != 7:
        raise SystemExit("internal passing-selection pin is malformed")
    expected_selection = {
        "candidate_index": int(fields[0]),
        "l1_coefficient": float(fields[1]),
        "auxiliary_coefficient": float(fields[2]),
        "activation_threshold": float(fields[3]),
        "validation_l0_mean": float(fields[4]),
        "validation_fvu_mean": float(fields[5]),
    }
    if result.get("selected_validation_configuration") != expected_selection:
        raise SystemExit("screening selection changed from its exact frozen value")
    selected_checkpoint = result.get("selected_checkpoint")
    checkpoint_sha = fields[6]
    expected_checkpoint_path = (
        Path(result_path).parent
        / "candidates"
        / f"candidate_{expected_selection['candidate_index']:03d}"
        / "best.pt"
    )
    if selected_checkpoint != {
        "path": str(expected_checkpoint_path),
        "sha256": checkpoint_sha,
    }:
        raise SystemExit("screening selected-checkpoint identity changed")
expected_manifest = {
    "schema_version": "r2_dictionary_control_run_manifest_v2",
    "status": status,
    "stage": "screening",
    "method": method,
    "model_name": model,
    "run_seed": 20260717,
    "executed_profile_sha256": profile_sha,
    "cache_manifest_sha256": cache_manifest_sha,
    "cache_content_sha256": cache_content_sha,
    "script_sha256": runner_sha,
    "module_sha256": module_sha,
    "result_sha256": result_sha,
    "checkpoint_sha256": checkpoint_sha,
}
if any(manifest.get(key) != value for key, value in expected_manifest.items()):
    raise SystemExit(f"screening manifest contract mismatch: {model}/{method}")
PY
  if [ "${status}" = completed_validation_screening ]; then
    candidate_index="${selection%%,*}"
    checkpoint_sha="${selection##*,}"
    checkpoint="${directory}/candidates/candidate_$(printf '%03d' "${candidate_index}")/best.pt"
    require_sha "${checkpoint_sha}" "${checkpoint}"
  fi
}

verify_screening_panel() {
  verify_screening protgpt2 relu_l1_sae
  verify_screening protgpt2 gated_sae
  verify_screening zymctrl relu_l1_sae
  verify_screening zymctrl gated_sae
  verify_screening progen2-medium relu_l1_sae
  verify_screening progen2-medium gated_sae
}

queue_runs() {
  case "${QUEUE_ID}" in
    0)
      printf '%s\n' \
        "protgpt2 topk_clt 17" \
        "protgpt2 dense_low_rank 17" \
        "zymctrl topk_clt 17" \
        "zymctrl gated_sae 17" \
        "zymctrl dense_low_rank 17" \
        "progen2-medium gated_sae 17" \
        "progen2-medium dense_low_rank 17"
      ;;
    1)
      printf '%s\n' \
        "protgpt2 topk_clt 29" \
        "protgpt2 dense_low_rank 29" \
        "zymctrl topk_clt 29" \
        "zymctrl gated_sae 29" \
        "zymctrl dense_low_rank 29" \
        "progen2-medium gated_sae 29" \
        "progen2-medium dense_low_rank 29"
      ;;
    2)
      printf '%s\n' \
        "protgpt2 topk_clt 43" \
        "protgpt2 dense_low_rank 43" \
        "zymctrl topk_clt 43" \
        "zymctrl gated_sae 43" \
        "zymctrl dense_low_rank 43" \
        "progen2-medium gated_sae 43" \
        "progen2-medium dense_low_rank 43"
      ;;
    3)
      printf '%s\n' \
        "zymctrl relu_l1_sae 17" \
        "zymctrl relu_l1_sae 29" \
        "zymctrl relu_l1_sae 43" \
        "progen2-medium topk_clt 17" \
        "progen2-medium topk_clt 29" \
        "progen2-medium topk_clt 43"
      ;;
  esac
}

screening_result_for_run() {
  case "$1/$2" in
    zymctrl/relu_l1_sae|zymctrl/gated_sae|progen2-medium/gated_sae)
      local result_sha manifest_sha status selection
      read -r result_sha manifest_sha status selection \
        < <(screening_identity "$1" "$2")
      test "${status}" = completed_validation_screening
      echo "${SCREENING_ROOT}/$1/$2/results.json ${result_sha}"
      ;;
    */topk_clt|*/dense_low_rank) return 1 ;;
    *)
      echo "run matrix contains an ineligible screened method: $1/$2" >&2
      return 2
      ;;
  esac
}

candidate_index_for_run() {
  case "$1/$2" in
    zymctrl/gated_sae) echo 3 ;;
    *) echo 0 ;;
  esac
}

output_for_run() {
  echo "${OUTPUT_ROOT}/$1/$2/seed_$3"
}

verify_in_progress() {
  local output="$1"
  local model="$2"
  local method="$3"
  local seed="$4"
  local content_sha="$5"
  local candidate_index="$6"
  local progress

  progress="${output}/candidates/candidate_$(printf '%03d' "${candidate_index}")/progress.pt"
  test -f "${progress}" || {
    echo "in-progress run lacks its exact progress checkpoint: ${output}" >&2
    exit 2
  }
  python3 - "${output}/run_state.json" "${model}" "${method}" "${seed}" \
    "${content_sha}" "${PROFILE_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema_version": "r2_dictionary_control_run_state_v1",
    "status": "in_progress",
    "stage": "full",
    "method": sys.argv[3],
    "model_name": sys.argv[2],
    "run_seed": int(sys.argv[4]),
    "cache_content_sha256": sys.argv[5],
    "profile_sha256": sys.argv[6],
}
if state != expected:
    raise SystemExit("in-progress run-state identity mismatch")
PY
}

verify_terminal() {
  local output="$1"
  local model="$2"
  local method="$3"
  local seed="$4"
  local cache_manifest_sha="$5"
  local cache_content_sha="$6"
  local screening_path="none"
  local screening_sha="none"
  local candidate_index

  candidate_index="$(candidate_index_for_run "${model}" "${method}")"
  case "${method}" in
    relu_l1_sae|gated_sae)
      if ! read -r screening_path screening_sha \
        < <(screening_result_for_run "${model}" "${method}"); then
        echo "screened full run lacks an eligible frozen selection: ${model}/${method}" >&2
        exit 2
      fi
      ;;
    topk_clt|dense_low_rank) ;;
    *) return 2 ;;
  esac
  python3 - "${output}/results.json" "${output}/run_manifest.json" \
    "${output}/run_state.json" "${output}" "${model}" "${method}" "${seed}" \
    "${cache_manifest_sha}" "${cache_content_sha}" "${PROFILE_SHA256}" \
    "${RUNNER_SHA256}" "${MODULE_SHA256}" "${screening_path}" \
    "${screening_sha}" "${candidate_index}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    result_name,
    manifest_name,
    state_name,
    output_name,
    model,
    method,
    seed_text,
    cache_manifest_sha,
    cache_content_sha,
    profile_sha,
    runner_sha,
    module_sha,
    screening_name,
    screening_sha,
    candidate_index_text,
) = sys.argv[1:]
result_path, manifest_path, state_path = map(
    Path, (result_name, manifest_name, state_name)
)
output = Path(output_name)
seed = int(seed_text)
candidate_index = int(candidate_index_text)
result = json.loads(result_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
state = json.loads(state_path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected_result = {
    "schema_version": "r2_dictionary_control_results_v1",
    "status": "completed_confirmatory_control",
    "p0_2_eligible": None,
    "quality_gate_requires_all_seed_aggregation": True,
    "stage": "full",
    "method": method,
    "model_name": model,
    "run_seed": seed,
    "profile_sha256": profile_sha,
    "cache_content_sha256": cache_content_sha,
    "test_evaluation_count": 1,
}
if any(result.get(key) != value for key, value in expected_result.items()):
    raise SystemExit("terminal full result violates the frozen contract")
if not isinstance(result.get("heldout_test"), dict):
    raise SystemExit("terminal full result lacks its one held-out test report")
if len(result.get("candidate_validation", [])) != 1:
    raise SystemExit("terminal full result must contain exactly one candidate")
candidate = result["candidate_validation"][0]
expected_candidate_id = (
    f"full_{method}_seed_{seed}_candidate_{candidate_index:03d}_"
    f"profile_{profile_sha[:12]}_cache_{cache_content_sha[:12]}"
)
if candidate.get("candidate_id") != expected_candidate_id:
    raise SystemExit("terminal full candidate identity mismatch")
checkpoint = output / "candidates" / f"candidate_{candidate_index:03d}" / "best.pt"
validation = checkpoint.with_name("validation_result.json")
if not checkpoint.is_file() or not validation.is_file():
    raise SystemExit("terminal full candidate lacks a retained best/validation file")
checkpoint_sha = sha256(checkpoint)
expected_checkpoint = {
    "path": str(checkpoint),
    "sha256": checkpoint_sha,
}
if result.get("selected_checkpoint") != expected_checkpoint:
    raise SystemExit("terminal selected-checkpoint contract mismatch")
if candidate.get("best_checkpoint") != expected_checkpoint:
    raise SystemExit("terminal candidate best-checkpoint contract mismatch")
if json.loads(validation.read_text(encoding="utf-8")) != candidate:
    raise SystemExit("retained validation result differs from the terminal candidate")
if screening_name == "none":
    if result.get("screening_result_sha256") is not None:
        raise SystemExit("unscreened full run consumed a screening result")
    expected_frozen = {
        "candidate_index": 0,
        "l1_coefficient": 0.0,
        "auxiliary_coefficient": 0.0,
        "activation_threshold": 0.0,
    }
else:
    screening_path = Path(screening_name)
    if sha256(screening_path) != screening_sha:
        raise SystemExit("full run's screening-result pin changed")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    expected_frozen = screening["selected_validation_configuration"]
    if result.get("screening_result_sha256") != screening_sha:
        raise SystemExit("full run reports a different screening-result hash")
if result.get("frozen_screening_configuration") != expected_frozen:
    raise SystemExit("full run changed its frozen validation selection")
selected = result.get("selected_validation_configuration", {})
for key in (
    "candidate_index",
    "l1_coefficient",
    "auxiliary_coefficient",
    "activation_threshold",
):
    if selected.get(key) != expected_frozen[key]:
        raise SystemExit("full run selected outside its frozen configuration")
result_sha = sha256(result_path)
expected_manifest = {
    "schema_version": "r2_dictionary_control_run_manifest_v2",
    "status": "completed_confirmatory_control",
    "stage": "full",
    "method": method,
    "model_name": model,
    "run_seed": seed,
    "executed_profile_sha256": profile_sha,
    "cache_manifest_sha256": cache_manifest_sha,
    "cache_content_sha256": cache_content_sha,
    "script_sha256": runner_sha,
    "module_sha256": module_sha,
    "result_sha256": result_sha,
    "checkpoint_sha256": checkpoint_sha,
}
if any(manifest.get(key) != value for key, value in expected_manifest.items()):
    raise SystemExit("terminal full manifest violates the frozen contract")
expected_state = {
    "schema_version": "r2_dictionary_control_run_state_v1",
    "status": "completed_confirmatory_control",
    "stage": "full",
    "method": method,
    "model_name": model,
    "run_seed": seed,
    "cache_content_sha256": cache_content_sha,
    "profile_sha256": profile_sha,
}
if state != expected_state:
    raise SystemExit("terminal full run-state violates the frozen contract")
PY
}

preflight_one() {
  local model="$1"
  local method="$2"
  local seed="$3"
  local output cache_manifest_sha cache_receipt_sha cache_content_sha payload_bytes
  local candidate_index

  output="$(output_for_run "${model}" "${method}" "${seed}")"
  read -r cache_manifest_sha cache_receipt_sha cache_content_sha payload_bytes \
    < <(cache_identity "${model}")
  candidate_index="$(candidate_index_for_run "${model}" "${method}")"
  if [ "${QUEUE_MODE}" = fresh ]; then
    if [ -e "${output}" ]; then
      echo "fresh mode refuses existing queue output: ${output}" >&2
      exit 2
    fi
    return
  fi
  if [ ! -e "${output}" ]; then
    return
  fi
  if [ -f "${output}/results.json" ] && [ -f "${output}/run_manifest.json" ]; then
    verify_terminal "${output}" "${model}" "${method}" "${seed}" \
      "${cache_manifest_sha}" "${cache_content_sha}"
    return
  fi
  if [ -e "${output}/results.json" ] || [ -e "${output}/run_manifest.json" ]; then
    echo "partial terminal artifacts forbid resume: ${output}" >&2
    exit 2
  fi
  verify_in_progress "${output}" "${model}" "${method}" "${seed}" \
    "${cache_content_sha}" "${candidate_index}"
}

cleanup_progress() {
  local output="$1"
  local model="$2"
  local method="$3"
  local seed="$4"
  local candidate_index progress bytes log

  candidate_index="$(candidate_index_for_run "${model}" "${method}")"
  progress="${output}/candidates/candidate_$(printf '%03d' "${candidate_index}")/progress.pt"
  log="${OUTPUT_ROOT}/storage_cleanup_queue_${QUEUE_ID}.tsv"
  if [ -f "${progress}" ]; then
    bytes="$(stat -c '%s' "${progress}")"
    rm -- "${progress}"
    test ! -e "${progress}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${bytes}" "${progress}" \
      "${model}" "${method}" "${seed}" >>"${log}"
    echo "[$(date --iso-8601=seconds)] removed verified redundant progress checkpoint bytes=${bytes} path=${progress}"
  fi
}

run_one() {
  local model="$1"
  local method="$2"
  local seed="$3"
  local output cache_manifest_sha cache_receipt_sha cache_content_sha payload_bytes
  local candidate_index action gpu_processes screening_path screening_sha
  local -a command

  output="$(output_for_run "${model}" "${method}" "${seed}")"
  read -r cache_manifest_sha cache_receipt_sha cache_content_sha payload_bytes \
    < <(cache_identity "${model}")
  candidate_index="$(candidate_index_for_run "${model}" "${method}")"
  action=fresh
  if [ -e "${output}" ]; then
    if [ "${QUEUE_MODE}" != resume ]; then
      echo "fresh run output appeared after queue preflight: ${output}" >&2
      exit 2
    fi
    if [ -f "${output}/results.json" ] && [ -f "${output}/run_manifest.json" ]; then
      verify_terminal "${output}" "${model}" "${method}" "${seed}" \
        "${cache_manifest_sha}" "${cache_content_sha}"
      cleanup_progress "${output}" "${model}" "${method}" "${seed}"
      echo "[$(date --iso-8601=seconds)] verified and skipped complete full run model=${model} method=${method} seed=${seed}"
      return
    fi
    if [ -e "${output}/results.json" ] || [ -e "${output}/run_manifest.json" ]; then
      echo "partial terminal artifacts appeared after queue preflight: ${output}" >&2
      exit 2
    fi
    verify_in_progress "${output}" "${model}" "${method}" "${seed}" \
      "${cache_content_sha}" "${candidate_index}"
    action=resume
  fi
  gpu_processes="$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid \
    --format=csv,noheader,nounits)"
  if grep -Eq '[0-9]' <<<"${gpu_processes}"; then
    echo "GPU ${GPU_ID} is occupied before ${model}/${method}/seed_${seed}" >&2
    exit 2
  fi

  command=(
    python3 "${RUNNER}"
    --cache-manifest "${CACHE_ROOT}/${model}/manifest.json"
    --cache-sha256 "${cache_manifest_sha}"
    --output-dir "${output}"
    --seed "${seed}"
    --device "cuda:${GPU_ID}"
    --method "${method}"
    --stage full
    --model-name "${model}"
    --profile "${PROFILE}"
    --profile-sha256 "${PROFILE_SHA256}"
    --pod-name "${POD_NAME}"
    --node-name "${NODE_NAME}"
    --gpu-index "${GPU_ID}"
    --git-commit "${GIT_COMMIT}"
    --git-dirty true
  )
  case "${method}" in
    relu_l1_sae|gated_sae)
      if ! read -r screening_path screening_sha \
        < <(screening_result_for_run "${model}" "${method}"); then
        echo "screened full run lacks an eligible frozen selection: ${model}/${method}" >&2
        exit 2
      fi
      command+=(
        --screening-result "${screening_path}"
        --screening-result-sha256 "${screening_sha}"
      )
      ;;
    topk_clt|dense_low_rank) ;;
    *) return 2 ;;
  esac
  if [ "${action}" = resume ]; then
    command+=(--resume)
  fi

  echo "[$(date --iso-8601=seconds)] starting stage=full mode=${action} model=${model} method=${method} seed=${seed} gpu=${GPU_ID} cache_manifest_sha256=${cache_manifest_sha} archive_sha256=${CODE_ARCHIVE_SHA256} launcher_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  free -h
  "${command[@]}"
  verify_terminal "${output}" "${model}" "${method}" "${seed}" \
    "${cache_manifest_sha}" "${cache_content_sha}"
  cleanup_progress "${output}" "${model}" "${method}" "${seed}"
  echo "[$(date --iso-8601=seconds)] completed stage=full model=${model} method=${method} seed=${seed} gpu=${GPU_ID} results_sha256=$(sha256sum "${output}/results.json" | awk '{print $1}') run_manifest_sha256=$(sha256sum "${output}/run_manifest.json" | awk '{print $1}')"
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  free -h
}

verify_deployment
verify_cache_receipt protgpt2
verify_cache_receipt zymctrl
verify_cache_receipt progen2-medium
verify_screening_panel
mkdir -p "${OUTPUT_ROOT}"

while read -r model method seed; do
  preflight_one "${model}" "${method}" "${seed}"
done < <(queue_runs)

while read -r model method seed; do
  run_one "${model}" "${method}" "${seed}"
done < <(queue_runs)

echo "[$(date --iso-8601=seconds)] full queue complete queue=${QUEUE_ID} gpu=${GPU_ID} mode=${QUEUE_MODE}"
