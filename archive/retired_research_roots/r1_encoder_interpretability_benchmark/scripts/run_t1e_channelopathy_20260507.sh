#!/usr/bin/env bash
set -euo pipefail

cd /oss-pvc/zhk_zip/biocc

export PYTHONUNBUFFERED=1
mkdir -p r1_encoder_interpretability_benchmark/logs/runtime

python3 r1_encoder_interpretability_benchmark/scripts/31_channelopathy_concordance.py \
  --labels r1_encoder_interpretability_benchmark/data/channelopathy/channelopathy_mechanism_positive_labels.tsv \
  --swissprot-fasta r1_encoder_interpretability_benchmark/data/channelopathy/channelopathy_canonical_sequences.fasta \
  --esm-model /gpfs/jiaotongdamoxing/zhk_zip/models/esm2_t36_3B_UR50D \
  --gpu 0 \
  --out-prefix r1_encoder_interpretability_benchmark/results/variant_effect/channelopathy_concordance_20260507
