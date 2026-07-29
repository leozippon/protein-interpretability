#!/usr/bin/env bash
set -euo pipefail
export BIOCC_EXTERNAL_RESOURCES=/oss-pvc/zhk_zip/biocc/external_resources
export PATH="$BIOCC_EXTERNAL_RESOURCES/tools/bin:$PATH"
export BIOCC_PFAM_A_HMM="$BIOCC_EXTERNAL_RESOURCES/ec_metrics/pfam/Pfam-A.hmm"
export BIOCC_ALPHAMISSENSE_HG38="$BIOCC_EXTERNAL_RESOURCES/baselines/alphamissense/AlphaMissense_hg38.tsv.gz"
export BIOCC_ALPHAMISSENSE_AA="$BIOCC_EXTERNAL_RESOURCES/baselines/alphamissense/AlphaMissense_aa_substitutions.tsv.gz"
export BIOCC_ESM1V_CHECKPOINT_DIR="$BIOCC_EXTERNAL_RESOURCES/baselines/esm1v/checkpoints"
export BIOCC_GMVP_HG38="$BIOCC_EXTERNAL_RESOURCES/baselines/gmvp/gMVP.2021-02-28.csv.gz"
export BIOCC_DBNSFP_GRCH38="$BIOCC_EXTERNAL_RESOURCES/baselines/dbnsfp/dbNSFP5.3.1a_grch38.gz"
export BIOCC_DBNSFP_GRCH38_TBI="$BIOCC_EXTERNAL_RESOURCES/baselines/dbnsfp/dbNSFP5.3.1a_grch38.gz.tbi"
export BIOCC_DBNSFP_GENE="$BIOCC_EXTERNAL_RESOURCES/baselines/dbnsfp/dbNSFP5.3_gene.gz"
export BIOCC_CLEAN_ROOT="$BIOCC_EXTERNAL_RESOURCES/ec_metrics/clean/CLEAN"
export BIOCC_CLEAN_ESM1B="$BIOCC_EXTERNAL_RESOURCES/ec_metrics/clean/esm1b_checkpoints/esm1b_t33_650M_UR50S.pt"
export BIOCC_CLEAN_ESM1B_CONTACT="$BIOCC_EXTERNAL_RESOURCES/ec_metrics/clean/esm1b_checkpoints/esm1b_t33_650M_UR50S-contact-regression.pt"
export BIOCC_CLEAN_PRETRAINED="$BIOCC_CLEAN_ROOT/app/data/pretrained"
export BIOCC_FOLDSEEK_PDB100_ARCHIVE="$BIOCC_EXTERNAL_RESOURCES/ec_metrics/foldseek/pdb100_20240101.tar.gz"
export BIOCC_FOLDSEEK_PDB100_VERSION="$BIOCC_EXTERNAL_RESOURCES/ec_metrics/foldseek/pdb100_20240101.version"
