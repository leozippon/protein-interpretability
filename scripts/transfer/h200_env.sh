#!/usr/bin/env bash
# Pod-side resource locations. This file is sourced, so it must not change the caller's shell options.

TRANSFER_GPFS_ROOT=/gpfs/jiaotongdamoxing/zhk_zip
TRANSFER_PROJECT_ROOT="${TRANSFER_GPFS_ROOT}/InterpretabilityTransfer"
TRANSFER_TEXT_DATA_ROOT="${TRANSFER_GPFS_ROOT}/biocc/external_resources/transfer_gap"

export TRANSFER_REPO_ROOT="${TRANSFER_GPFS_ROOT}"
export TRANSFER_MODEL_BASE_DIR="${TRANSFER_GPFS_ROOT}/models"
export TRANSFER_TEXT_MODEL_DIR="${TRANSFER_MODEL_BASE_DIR}/gpt2-large"
export TRANSFER_TEXT_MODEL_BASE_DIR="${TRANSFER_MODEL_BASE_DIR}"

export TRANSFER_OPENWEBTEXT_DIR="${TRANSFER_TEXT_DATA_ROOT}/openwebtext-screen/plain_text"
export TRANSFER_SWISSPROT_FASTA="${TRANSFER_GPFS_ROOT}/data/swissprot/uniprot_sprot.fasta.gz"
export TRANSFER_ZYMCTRL_FASTA="${TRANSFER_GPFS_ROOT}/data/zymctrl/ec_labeled_swissprot.fasta"
export TRANSFER_PFAM_RESIDUE_TSV="${TRANSFER_GPFS_ROOT}/data/interpro/pfam_residue.tsv"
export TRANSFER_ALPHAFOLD_DIR="${TRANSFER_GPFS_ROOT}/data/alphafold"
export TRANSFER_PROTEINGYM_DIR="${TRANSFER_GPFS_ROOT}/data/proteingym/DMS_ProteinGym_substitutions"
export TRANSFER_UNIREF50_FASTA="${TRANSFER_GPFS_ROOT}/data/uniref50/uniref50.fasta"

export TRANSFER_DIAMOND_TARBALL="/oss-pvc/zhk_zip/biocc/external_resources/tools/diamond-linux64-v2.1.24.tar.gz"
export TRANSFER_DIAMOND_CHECKSUM="/oss-pvc/zhk_zip/biocc/external_resources/tools/diamond-linux64-v2.1.24.tar.gz.sha256"
export TRANSFER_DIAMOND_DIR="${TRANSFER_PROJECT_ROOT}/resources/tools/diamond"
export TRANSFER_DIAMOND_DB="${TRANSFER_PROJECT_ROOT}/resources/homology/uniref50_full.dmnd"
export TRANSFER_DIAMOND_TMPDIR="${TRANSFER_PROJECT_ROOT}/scratch/diamond"

export TRANSFER_PYTHON=/opt/ac2/bin/python3
TRANSFER_PACKAGE_ROOT="${TRANSFER_PACKAGE_ROOT:-${TRANSFER_PROJECT_ROOT}/packages}"
export TRANSFER_PACKAGE_ROOT
export PYTHONPATH="${TRANSFER_PACKAGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${TRANSFER_PROJECT_ROOT}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
