#!/usr/bin/env bash
set -euo pipefail

# Run one fixed GPU queue from the validation-only P0-2 sparse screening panel.

ROOT="/gpfs/jiaotongdamoxing/zhk_zip/biocc/npj_revision_20260717"
PROJECT="${ROOT}/code_p0_2_dictionary_controls_bf16_r1/r2_interpretability_transfer"
CODE_ARCHIVE="${ROOT}/code_archives/r2_p0_2_dictionary_controls_bf16_payload_r1.tar"
TREE_MANIFEST="${PROJECT}/CODE_CONTENT_SHA256SUMS"
RUNNER="${PROJECT}/scripts/58_run_dictionary_controls.py"
PROFILE="${PROJECT}/configs/p0_2_dictionary_controls_production_profile.json"
CACHE_ROOT="${ROOT}/p0_2_exact_cache_bf16_r3"
OUTPUT_ROOT="${ROOT}/p0_2_dictionary_controls_bf16_r1/screening"
CODE_ARCHIVE_SHA256="${CODE_ARCHIVE_SHA256:?set the deployed code archive SHA-256}"
PROFILE_SHA256="eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2"
RUNNER_SHA256="99f4224f833dfc644f9adc45b2432b4c53a81b2cd857081d3386fc77ec513d8a"
GPU_ID="${GPU_ID:?set GPU_ID to 0, 1, 2 or 3}"
QUEUE_ID="${QUEUE_ID:?set QUEUE_ID to the matching fixed queue}"
POD_NAME="damoxing-zhk-zipbio-master-0"
NODE_NAME="i-d5cvmv6heob1nidq4ujg"
GIT_COMMIT="d765d4bf761d566fb45b6318c6067c9981cf96ad"
SCREENING_SEED=20260717

export PYTHONDONTWRITEBYTECODE=1

if [[ ! "${GPU_ID}" =~ ^[0-3]$ ]] || [ "${QUEUE_ID}" != "${GPU_ID}" ]; then
  echo "GPU_ID and QUEUE_ID must be the same integer in 0..3" >&2
  exit 2
fi

require_sha() {
  local expected="$1"
  local path="$2"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

verify_deployment() {
  local archived_manifest_sha
  local deployed_manifest_sha

  require_sha "${CODE_ARCHIVE_SHA256}" "${CODE_ARCHIVE}"
  require_sha "${PROFILE_SHA256}" "${PROFILE}"
  require_sha "${RUNNER_SHA256}" "${RUNNER}"
  test -f "${TREE_MANIFEST}"
  archived_manifest_sha="$({
    tar -xOf "${CODE_ARCHIVE}" \
      r2_interpretability_transfer/CODE_CONTENT_SHA256SUMS
  } | sha256sum | awk '{print $1}')"
  deployed_manifest_sha="$(sha256sum "${TREE_MANIFEST}" | awk '{print $1}')"
  test "${archived_manifest_sha}" = "${deployed_manifest_sha}"
  if find "${PROJECT}" -type l -print -quit | grep -q .; then
    echo "deployed screening tree contains a symbolic link" >&2
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

verify_cache_receipt() {
  local model="$1"
  local manifest_sha="$2"
  local receipt_sha="$3"
  local payload_bytes="$4"
  local receipt="${CACHE_ROOT}/${model}/completion_receipt.json"
  local manifest="${CACHE_ROOT}/${model}/manifest.json"

  require_sha "${receipt_sha}" "${receipt}"
  require_sha "${manifest_sha}" "${manifest}"
  python3 - "${receipt}" "${manifest}" "${model}" "${PROFILE_SHA256}" \
    "${payload_bytes}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, manifest_path = map(Path, sys.argv[1:3])
model, profile_sha, payload_bytes = sys.argv[3], sys.argv[4], int(sys.argv[5])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

report_path = Path(receipt["execution_report_path"])
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
if not report_path.is_file() or sha256(report_path) != receipt["execution_report_sha256"]:
    raise SystemExit("cache receipt execution-report hash mismatch")
if manifest.get("content_sha256") != receipt["cache_content_sha256"]:
    raise SystemExit("cache receipt content identity mismatch")
PY
}

cache_identity() {
  case "$1" in
    protgpt2)
      echo "d3fa68212612bb42ed9d75c0b49a606db44a9171c00b0662c05b41f48f63dd34 c5e323df9618c14db9ec4e19332dc1b162a4a026b68381b2b1fd3c7954a73225 221184000000"
      ;;
    zymctrl)
      echo "26307ed3694c884543d3033cd99e6cee52cd5441c9ce7f37f6f9a4f1a48dbc70 0b4c4e9040556d73dee68421cff13cc1497e4c8348e03a4c72f9eb0bfd0fb013 221184000000"
      ;;
    progen2-medium)
      echo "e51c9dacfc7cbb56fb8f860800a4e2315686838853e34f1905d95fa757d72ea9 31b34aef4a059b061b4979a5afa5a4b07ccc0278b37ecac3e2a69d7153171a80 199065600000"
      ;;
    *) return 2 ;;
  esac
}

run_one() {
  local model="$1"
  local method="$2"
  local manifest_sha receipt_sha payload_bytes
  local output="${OUTPUT_ROOT}/${model}/${method}"
  local gpu_processes

  read -r manifest_sha receipt_sha payload_bytes < <(cache_identity "${model}")
  verify_cache_receipt \
    "${model}" "${manifest_sha}" "${receipt_sha}" "${payload_bytes}"
  if [ -e "${output}" ]; then
    echo "refusing to reuse screening output: ${output}" >&2
    exit 2
  fi
  gpu_processes="$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid \
    --format=csv,noheader,nounits)"
  if grep -Eq '[0-9]' <<<"${gpu_processes}"; then
    echo "GPU ${GPU_ID} is occupied before ${model}/${method}" >&2
    exit 2
  fi

  echo "[$(date --iso-8601=seconds)] starting stage=screening model=${model} method=${method} gpu=${GPU_ID} cache_manifest_sha256=${manifest_sha} cache_receipt_sha256=${receipt_sha} archive_sha256=${CODE_ARCHIVE_SHA256} launcher_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  free -h
  python3 "${RUNNER}" \
    --cache-manifest "${CACHE_ROOT}/${model}/manifest.json" \
    --cache-sha256 "${manifest_sha}" \
    --output-dir "${output}" \
    --seed "${SCREENING_SEED}" \
    --device "cuda:${GPU_ID}" \
    --method "${method}" \
    --stage screening \
    --model-name "${model}" \
    --profile "${PROFILE}" \
    --profile-sha256 "${PROFILE_SHA256}" \
    --pod-name "${POD_NAME}" \
    --node-name "${NODE_NAME}" \
    --gpu-index "${GPU_ID}" \
    --git-commit "${GIT_COMMIT}" \
    --git-dirty true
  test -f "${output}/results.json"
  test -f "${output}/run_manifest.json"
  python3 - "${output}/results.json" "${model}" "${method}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("status") not in {
    "completed_validation_screening",
    "sparsity_match_failure",
}:
    raise SystemExit("screening did not publish a terminal result")
if (
    result.get("model_name") != sys.argv[2]
    or result.get("method") != sys.argv[3]
    or result.get("run_seed") != 20260717
    or result.get("test_evaluation_count") != 0
    or result.get("p0_2_eligible") is not False
):
    raise SystemExit("screening terminal result violates the frozen contract")
PY
  echo "[$(date --iso-8601=seconds)] completed stage=screening model=${model} method=${method} gpu=${GPU_ID} results_sha256=$(sha256sum "${output}/results.json" | awk '{print $1}') run_manifest_sha256=$(sha256sum "${output}/run_manifest.json" | awk '{print $1}')"
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  free -h
}

verify_deployment
mkdir -p "${OUTPUT_ROOT}"

case "${QUEUE_ID}" in
  0) run_one protgpt2 gated_sae ;;
  1) run_one zymctrl gated_sae ;;
  2)
    run_one progen2-medium gated_sae
    run_one progen2-medium relu_l1_sae
    ;;
  3)
    run_one protgpt2 relu_l1_sae
    run_one zymctrl relu_l1_sae
    ;;
esac

echo "[$(date --iso-8601=seconds)] screening queue complete queue=${QUEUE_ID} gpu=${GPU_ID}"
