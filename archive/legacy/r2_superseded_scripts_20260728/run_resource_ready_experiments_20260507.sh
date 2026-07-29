#!/usr/bin/env bash
set -u

REPO=${BIOCC_REPO:-/oss-pvc/zhk_zip/biocc}
cd "$REPO" || exit 1

export BIOCC_EXTERNAL_RESOURCES=${BIOCC_EXTERNAL_RESOURCES:-/oss-pvc/zhk_zip/biocc/external_resources}
export PATH="$BIOCC_EXTERNAL_RESOURCES/tools/bin:$BIOCC_EXTERNAL_RESOURCES/tools/hmmer-3.4-install/bin:$PATH"
export BIOCC_GMVP_HG38=${BIOCC_GMVP_HG38:-$BIOCC_EXTERNAL_RESOURCES/baselines/gmvp/gMVP.2021-02-28.csv.gz}
export BIOCC_ESM1V_CHECKPOINT_DIR=${BIOCC_ESM1V_CHECKPOINT_DIR:-$BIOCC_EXTERNAL_RESOURCES/baselines/esm1v/checkpoints}
export BIOCC_CLEAN_ROOT=${BIOCC_CLEAN_ROOT:-$BIOCC_EXTERNAL_RESOURCES/ec_metrics/clean/CLEAN}
export BIOCC_CLEAN_ESM1B=${BIOCC_CLEAN_ESM1B:-$BIOCC_EXTERNAL_RESOURCES/ec_metrics/clean/esm1b_checkpoints/esm1b_t33_650M_UR50S.pt}
export BIOCC_FOLDSEEK=${BIOCC_FOLDSEEK:-$BIOCC_EXTERNAL_RESOURCES/tools/foldseek/bin/foldseek}
export BIOCC_FOLDSEEK_PDB100_ARCHIVE=${BIOCC_FOLDSEEK_PDB100_ARCHIVE:-$BIOCC_EXTERNAL_RESOURCES/ec_metrics/foldseek/pdb100_20240101.tar.gz}
export BIOCC_FOLDSEEK_PDB100_DIR=${BIOCC_FOLDSEEK_PDB100_DIR:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/external_resources/foldseek/pdb100_20240101}
export BIOCC_SWISSPROT_CACHE=${BIOCC_SWISSPROT_CACHE:-/gpfs/jiaotongdamoxing/zhk_zip/data/processed/swissprot_all_max1022.pkl}

mkdir -p r1_encoder_interpretability_benchmark/logs/runtime r2_decoder_sparse_readout_audit/logs/runtime

run_step() {
  local name="$1"
  shift
  echo
  echo "===== START $name $(date -Is) ====="
  "$@"
  local status=$?
  echo "===== END $name status=$status $(date -Is) ====="
  return 0
}

echo "Resource-ready TODO runner"
echo "repo=$REPO"
echo "host=$(hostname)"
date -Is
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
python3 -c 'import esm; print("fair-esm import ok", esm.__file__)' || true

run_step "R1 external baselines gMVP+ESM1v" \
  python3 r1_encoder_interpretability_benchmark/scripts/28_external_baselines_available.py \
    --esm1v-checkpoint-limit 5 \
    --n-bootstrap 1000 \
    --out-json r1_encoder_interpretability_benchmark/results/variant_effect/external_baselines_available_20260507.json \
    --out-md r1_encoder_interpretability_benchmark/results/variant_effect/external_baselines_available_20260507.md \
    --out-tsv r1_encoder_interpretability_benchmark/results/variant_effect/external_baselines_available_scores_20260507.tsv

run_step "R2 CLEAN generated EC prediction" \
  python3 r2_decoder_sparse_readout_audit/scripts/19_clean_generated.py \
    --out-json r2_decoder_sparse_readout_audit/results/ec_metrics/clean_generated_lysozyme_20260507.json

run_step "R2 Foldseek generated structure scan" \
  python3 r2_decoder_sparse_readout_audit/scripts/18_foldseek_generated.py \
    --threads 16 \
    --max-seqs 20 \
    --out-json r2_decoder_sparse_readout_audit/results/ec_metrics/foldseek_generated_lysozyme_20260507.json

echo
echo "All queued resource-ready steps attempted."
date -Is
