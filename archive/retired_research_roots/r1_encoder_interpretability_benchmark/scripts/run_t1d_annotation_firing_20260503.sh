#!/usr/bin/env bash
set -euo pipefail

echo "T1-D annotation firing rerun started $(date '+%F %T %Z')"

RESEARCH1_ROOT="${RESEARCH1_ROOT:-/oss-pvc/zhk_zip/biocc/Research1}"
REPO_ROOT="$(dirname "${RESEARCH1_ROOT}")"
CKPT_ROOT="${CKPT_ROOT:-/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research1/results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights}"
ESM_MODEL="${ESM_MODEL:-/gpfs/jiaotongdamoxing/zhk_zip/models/esm2_t36_3B_UR50D}"
SWISSPROT_CACHE="${SWISSPROT_CACHE:-/gpfs/jiaotongdamoxing/zhk_zip/data/processed/swissprot_all_max1022.pkl}"
GO_GAF="${GO_GAF:-/gpfs/jiaotongdamoxing/zhk_zip/data/go/goa_uniprot_all.gaf.gz}"
PFAM_TSV="${PFAM_TSV:-/oss-pvc/zhk_zip/biocc/data/interpro/pfam_residue.tsv}"
BIOLIP="${BIOLIP:-/oss-pvc/zhk_zip/biocc/data/BioLiP/BioLiP.txt}"
NUM_PROTEINS="${NUM_PROTEINS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_FIRING="${MAX_FIRING:-200}"
LAYERS="${LAYERS:-19 23 27 31 35}"

cd "${RESEARCH1_ROOT}"
mkdir -p results/annotation_alignment logs/runtime

for layer in ${LAYERS}; do
  echo "===== L${layer} $(date '+%F %T %Z') ====="
  pkl="results/annotation_alignment/ours_3B_l${layer}_step500000.pkl"
  js="results/annotation_alignment/ours_3B_l${layer}_step500000_summary.json"
  if [[ -f "${pkl}" && ! -f "${pkl%.pkl}_nofiring_backup_20260503.pkl" ]]; then
    cp "${pkl}" "${pkl%.pkl}_nofiring_backup_20260503.pkl"
  fi
  if [[ -f "${js}" && ! -f "${js%.json}_nofiring_backup_20260503.json" ]]; then
    cp "${js}" "${js%.json}_nofiring_backup_20260503.json"
  fi

  CUDA_VISIBLE_DEVICES=0 python3 scripts/04_analyze_our_sae.py \
    --gpu 0 \
    --num-proteins "${NUM_PROTEINS}" \
    --batch-size "${BATCH_SIZE}" \
    --layer "${layer}" \
    --step 500000 \
    --checkpoint-root "${CKPT_ROOT}" \
    --esm-model "${ESM_MODEL}" \
    --cache-path "${SWISSPROT_CACHE}" \
    --save-firing-positions \
    --max-firing-positions-per-feature "${MAX_FIRING}" \
    --out-prefix "ours_3B_l${layer}_step500000"
done

cd "${REPO_ROOT}"
python3 r1_encoder_interpretability_benchmark/scripts/19_expand_annotation.py \
  --layers 19 23 27 31 35 \
  --annotation-dir r1_encoder_interpretability_benchmark/results/annotation_alignment \
  --go-gaf "${GO_GAF}" \
  --pfam-tsv "${PFAM_TSV}" \
  --biolip "${BIOLIP}" \
  --swissprot-cache "${SWISSPROT_CACHE}" \
  --out r1_encoder_interpretability_benchmark/results/annotation_alignment/expanded_summary_firing_20260503.json

echo "T1-D annotation firing rerun finished $(date '+%F %T %Z')"
