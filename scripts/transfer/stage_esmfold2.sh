#!/usr/bin/env bash
# Stage the isolated ESMFold2 runtime and the offline biohub/ESMFold2-hf weights.
# Run on Compute. Do not install into ct. Do not write credentials.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_DIR="${TRANSFER_FOLDING_RUNTIME:-$ROOT/runtimes/esmfold2}"
MODEL_DIR="${TRANSFER_MODEL_BASE_DIR:-$HOME/models}/ESMFold2-hf"
ENV_FILE="${ROOT}/.env.local"
CT_PYTHON="${CT_PYTHON:-/Data/lzp/miniconda3/envs/ct/bin/python}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "stage_esmfold2: HF_TOKEN is unset; copy .env.local.example to .env.local" >&2
  exit 1
fi
if [[ ! -x "$CT_PYTHON" ]]; then
  echo "stage_esmfold2: ct python is missing at $CT_PYTHON" >&2
  exit 1
fi

# Reuse ct's CUDA torch; overlay only Transformers 5.x in this venv.
"$CT_PYTHON" -m venv --system-site-packages "$RUNTIME_DIR"
# shellcheck disable=SC1091
source "$RUNTIME_DIR/bin/activate"
if [[ "$(command -v python)" != "$RUNTIME_DIR/bin/python" ]]; then
  echo "stage_esmfold2: venv python is not isolated" >&2
  exit 1
fi
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements-esmfold2.txt"
python - <<'PY'
import transformers
from transformers import EsmFold2Model

print("python", __import__("sys").version.split()[0])
print("torch", __import__("torch").__version__)
print("transformers", transformers.__version__)
print("EsmFold2Model", EsmFold2Model.__module__)
if tuple(int(part) for part in transformers.__version__.split(".")[:2]) < (5, 17):
    raise SystemExit(f"need transformers>=5.17.0, found {transformers.__version__}")
PY

mkdir -p "$MODEL_DIR"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# HF_TOKEN is already in the environment; do not put it on the command line.
if command -v hf >/dev/null; then
  hf download biohub/ESMFold2-hf --local-dir "$MODEL_DIR"
else
  python -m pip install huggingface_hub
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("biohub/ESMFold2-hf", local_dir="${MODEL_DIR}")
PY
fi

echo "runtime=$RUNTIME_DIR"
echo "model=$MODEL_DIR"
