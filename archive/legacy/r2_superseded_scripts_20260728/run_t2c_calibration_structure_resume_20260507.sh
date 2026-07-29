#!/usr/bin/env bash
set -euo pipefail

cd /oss-pvc/zhk_zip/biocc

export BIOCC_FOLDSEEK="/oss-pvc/zhk_zip/biocc/external_resources/tools/foldseek/bin/foldseek"
export BIOCC_FOLDSEEK_PDB100_ARCHIVE="/oss-pvc/zhk_zip/biocc/external_resources/ec_metrics/foldseek/pdb100_20240101.tar.gz"
export BIOCC_FOLDSEEK_PDB100_DIR="/gpfs/jiaotongdamoxing/zhk_zip/biocc/external_resources/foldseek/pdb100_20240101"
export ESMFOLD_PATH="/gpfs/jiaotongdamoxing/zhk_zip/models/esmfold_v1"

CAL_DIR="r2_decoder_sparse_readout_audit/results/ec_metrics/calibration_lysozyme_20260507"
GPFS_CAL_DIR="/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/ec_metrics/calibration_lysozyme_20260507"
REAL_PDB_DIR="${GPFS_CAL_DIR}/real_esmfold_pdbs"
RANDOM_PDB_DIR="${GPFS_CAL_DIR}/random_esmfold_pdbs"

mkdir -p "r2_decoder_sparse_readout_audit/logs/runtime" "${REAL_PDB_DIR}" "${RANDOM_PDB_DIR}"
echo "[structure-resume-start] $(date -Is)"

python3 r2_decoder_sparse_readout_audit/scripts/13_structural_qc.py \
  --input "${CAL_DIR}/calibration_real_lysozyme.json" \
  --field records \
  --max-fold "${BIOCC_CAL_MAX_FOLD:-100}" \
  --max-structures "${BIOCC_CAL_MAX_STRUCTURES:-100}" \
  --out r2_decoder_sparse_readout_audit/results/ec_metrics/calibration_real_lysozyme_esmfold_20260507.json \
  --pdb-dir "${REAL_PDB_DIR}" \
  --backend local \
  --dtype fp32

python3 r2_decoder_sparse_readout_audit/scripts/13_structural_qc.py \
  --input "${CAL_DIR}/calibration_random_uniref50.json" \
  --field records \
  --max-fold "${BIOCC_CAL_MAX_FOLD:-100}" \
  --max-structures "${BIOCC_CAL_MAX_STRUCTURES:-100}" \
  --out r2_decoder_sparse_readout_audit/results/ec_metrics/calibration_random_uniref50_esmfold_20260507.json \
  --pdb-dir "${RANDOM_PDB_DIR}" \
  --backend local \
  --dtype fp32

python3 r2_decoder_sparse_readout_audit/scripts/22_foldseek_calibration.py \
  --real-pdb-dir "${REAL_PDB_DIR}" \
  --random-pdb-dir "${RANDOM_PDB_DIR}" \
  --threads 16 \
  --out-json r2_decoder_sparse_readout_audit/results/ec_metrics/foldseek_calibration_lysozyme_20260507.json

python3 r2_decoder_sparse_readout_audit/scripts/23_ec_metric_calibration_summary.py \
  --out-json r2_decoder_sparse_readout_audit/results/ec_metrics/ec_metric_calibration_summary_20260507.json

echo "[structure-resume-done] $(date -Is)"
