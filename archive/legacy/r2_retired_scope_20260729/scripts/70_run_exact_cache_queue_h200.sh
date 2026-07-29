#!/usr/bin/env bash
set -euo pipefail

# Run the complete code-bound BF16 P0-2 activation-cache panel on GPU 3.

ROOT="/gpfs/jiaotongdamoxing/zhk_zip/biocc/npj_revision_20260717"
PROJECT="${ROOT}/code_p0_2_exact_cache_bf16_r8/r2_interpretability_transfer"
CODE_ARCHIVE="${ROOT}/code_archives/r2_p0_2_exact_cache_bf16_payload_r8.tar"
CODE_CONTENT_MANIFEST="${PROJECT}/CODE_CONTENT_SHA256SUMS"
RUNNER="${PROJECT}/scripts/61_build_dictionary_activation_cache.py"
PROFILE="${PROJECT}/configs/p0_2_dictionary_controls_production_profile.json"
PANEL="${ROOT}/p0_2_exact_cache_bf16_r3"
COHORTS="${ROOT}/cohorts"
MODELS="/gpfs/jiaotongdamoxing/zhk_zip/models"
CODE_ARCHIVE_SHA256="${CODE_ARCHIVE_SHA256:?set the deployed code archive SHA-256}"
CODE_CONTENT_MANIFEST_SHA256="${CODE_CONTENT_MANIFEST_SHA256:?set the deployed code-content manifest SHA-256}"
PROFILE_SHA256="${PROFILE_SHA256:?set the deployed production profile SHA-256}"
RUNNER_SHA256="${RUNNER_SHA256:?set the deployed cache runner SHA-256}"
GPU_ID=3
POD_NAME="damoxing-zhk-zipbio-master-0"
NODE_NAME="i-d5cvmv6heob1nidq4ujg"
GIT_COMMIT="d765d4bf761d566fb45b6318c6067c9981cf96ad"

export PYTHONDONTWRITEBYTECODE=1

require_sha() {
  local expected="$1"
  local path="$2"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

run_cache() {
  local model="$1"
  local model_dir="$2"
  local cohort="$3"
  local config_sha="$4"
  local weights_sha="$5"
  local tokenizer_sha="$6"
  local train_sha="$7"
  local validation_sha="$8"
  local test_sha="$9"
  local model_revision="${10}"
  local output="${PANEL}/${model}"
  local gpu_processes

  test ! -e "${output}"
  if ! gpu_processes="$(
    nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid \
      --format=csv,noheader,nounits
  )"; then
    echo "nvidia-smi process query failed for GPU ${GPU_ID}" >&2
    exit 2
  fi
  if grep -Eq '[0-9]' <<<"${gpu_processes}"; then
    echo "GPU ${GPU_ID} is occupied before ${model} cache extraction" >&2
    exit 2
  fi
  echo "[$(date --iso-8601=seconds)] starting model=${model} gpu=${GPU_ID}"
  echo "code_archive_sha256=${CODE_ARCHIVE_SHA256} code_content_manifest_sha256=${CODE_CONTENT_MANIFEST_SHA256} profile_sha256=${PROFILE_SHA256} runner_sha256=${RUNNER_SHA256}"
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  free -h
  python3 "${RUNNER}" \
    --model-name "${model}" \
    --model-root "${MODELS}/${model_dir}" \
    --profile "${PROFILE}" \
    --profile-sha256 "${PROFILE_SHA256}" \
    --code-archive "${CODE_ARCHIVE}" \
    --code-archive-sha256 "${CODE_ARCHIVE_SHA256}" \
    --code-content-manifest "${CODE_CONTENT_MANIFEST}" \
    --code-content-manifest-sha256 "${CODE_CONTENT_MANIFEST_SHA256}" \
    --output-dir "${output}" \
    --panel-cache-root "${PANEL}" \
    --device "cuda:${GPU_ID}" \
    --train-manifest "${COHORTS}/${cohort}/train.jsonl" \
    --train-manifest-sha256 "${train_sha}" \
    --validation-manifest "${COHORTS}/${cohort}/validation.jsonl" \
    --validation-manifest-sha256 "${validation_sha}" \
    --test-manifest "${COHORTS}/${cohort}/test.jsonl" \
    --test-manifest-sha256 "${test_sha}" \
    --model-revision "${model_revision}" \
    --model-config-sha256 "${config_sha}" \
    --model-weights-sha256 "${weights_sha}" \
    --tokenizer-sha256 "${tokenizer_sha}" \
    --pod-name "${POD_NAME}" \
    --node-name "${NODE_NAME}" \
    --gpu-index "${GPU_ID}" \
    --git-commit "${GIT_COMMIT}" \
    --git-dirty true
  test -f "${output}/completion_receipt.json"
  echo "[$(date --iso-8601=seconds)] completed model=${model} gpu=${GPU_ID}"
  if ! nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader; then
    echo "post-run nvidia-smi query failed for GPU ${GPU_ID}" >&2
    exit 2
  fi
  free -h
}

require_sha "${CODE_ARCHIVE_SHA256}" "${CODE_ARCHIVE}"
require_sha "${CODE_CONTENT_MANIFEST_SHA256}" "${CODE_CONTENT_MANIFEST}"
require_sha "${RUNNER_SHA256}" "${RUNNER}"
require_sha "${PROFILE_SHA256}" "${PROFILE}"

run_cache \
  protgpt2 ProtGPT2 uniref50 \
  b1eb50b2e360c8c9e433f234eeec53e809c99df254271ace5454b89244818195 \
  972767ca741ef0f31c165241e94692e3a5a799cfecc3bfe4d6db6938f1d229bc \
  b6899c546a4acc5cdbbfaac5471fa7b21d8bcb1148a7511612f257610701ec78 \
  f4270143c1c22904d4f548de9837572040d2b10b87d9c64363fdb811422525bb \
  8c3e9e83ce0d75ebaf0a4e1a0c95010b20875fc46fdfdeaf908d3290a5b207ae \
  d43737a0860579c52634d535449129bb04e2c3b02c37649d4a688e9f35d4de03 \
  huggingface-nferruz-ProtGPT2-f71aa6cf063ad784ebd53881d11332fd098eaa58

run_cache \
  zymctrl ZymCTRL zymctrl \
  e0ed9a64d87a2c29ae66d856d4ba3675bd78afac11a78fceb8ac2fe2d58b9c4f \
  5ebc23b19d63e352802552b9d7220be669ab6447d7a0a7ca427c3ae0873650fe \
  feb7088533715aae6b73e952d1f69b2c0f99ee62e3cd1f0615578ea22888a954 \
  f9727a374209f7329bed910624430fcf747b735314895215d1eff313b0d3f755 \
  f35fe72297bff551700a62c26e64094b4a23323faa701bfa5cc34b2ef239a0cf \
  a2b08d1980b78db955e6fc7558d9fa41c908dde597449c15cae803ede12f4289 \
  best-supported-huggingface-AI4PD-ZymCTRL-3c532ef172b9cd2e95238baadf5167ebb89fbc32-whole-tree-unresolved

run_cache \
  progen2-medium progen2-medium uniref50 \
  ed9358213db1b9449f02f0080483967b8bbab398a7fe4cae073523c717d924b1 \
  23aeeff4031fc56dd371dad5c408f6930dedfb6d46e538006bf7f589afc08c31 \
  cfc9cc1382e509ea9b77082e966e7a7374e1247cdd3681b2422ff1cdf3e190b1 \
  f4270143c1c22904d4f548de9837572040d2b10b87d9c64363fdb811422525bb \
  8c3e9e83ce0d75ebaf0a4e1a0c95010b20875fc46fdfdeaf908d3290a5b207ae \
  d43737a0860579c52634d535449129bb04e2c3b02c37649d4a688e9f35d4de03 \
  local-hybrid-deployment-no-single-upstream-commit

echo "[$(date --iso-8601=seconds)] exact-cache panel queue complete"
