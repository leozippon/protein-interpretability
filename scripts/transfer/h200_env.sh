#!/usr/bin/env bash
# Pod-side locations for the transfer measurement package.
#
# Source this before running anything under scripts/transfer/ inside an H200
# pod. Every variable here has a local-L20 default compiled into
# src/transfer/arms.py, so this file only states where the same inputs live on
# GPFS; it changes no measurement parameter, threshold, estimand or default.
#
# Each input is named separately rather than being derived from one root. The
# corpora sit under the shared data root but the checkpoints do not, and the
# text checkpoints do not sit with the protein ones, so a single root would only
# be honest about one of them. A path that is wrong here fails at first use with
# the variable named in the message (src.transfer.arms.require_input_path);
# nothing falls back.
#
# Sourcing this file does not verify that the paths exist. data/swissprot and
# data/alphafold are being staged, and a measurement that needs neither must
# still be able to run. Existence is the running measurement's check, not this
# file's.
#
# Usage:
#   source scripts/transfer/h200_env.sh
#   "$R2_PYTHON" scripts/transfer/01_cohort_power.py --arms zymctrl ...

# No `set -e`/`set -u` here: this file is sourced, so either would change the
# shell options of whatever sourced it.

# ---------------------------------------------------------------- GPFS layout
R2_GPFS_ROOT=/gpfs/jiaotongdamoxing/zhk_zip
R2_TRANSFER_GAP="${R2_GPFS_ROOT}/biocc/external_resources/transfer_gap"

# Repository and data root. data/uniref50, data/zymctrl, data/interpro,
# data/proteingym, data/swissprot and data/alphafold all resolve beneath it.
export R2_REPO_ROOT="${R2_GPFS_ROOT}"

# ----------------------------------------------------------------- checkpoints
# Protein decoders: ProtGPT2, ZymCTRL, progen2-*.
export R2_MODEL_BASE_DIR="${R2_GPFS_ROOT}/models"
# The text arm of the matched pair. This MUST equal
# ${R2_TEXT_MODEL_BASE_DIR}/gpt2-large: src.transfer.arms addresses gpt2-large
# through this variable while src.transfer.scaling's ladder addresses it by name
# under the base directory, and scaling.register_arm_spec refuses a ladder
# declaration whose path disagrees with the frozen panel declaration. Pointing
# the two at different trees does not produce a wrong number -- it produces a
# raised ValueError -- but it does cost a scheduled run, so they are kept
# adjacent and this comment says why.
export R2_TEXT_MODEL_DIR="${R2_GPFS_ROOT}/models/gpt2-large"
# Parent of the text checkpoints addressed by name: the GPT-2 convergence ladder,
# the ByGPT5 byte-level rungs, DialoGPT-small and the rotary arms. Rungs that are
# not staged are reported unavailable by src.transfer.scaling.inspect_member
# rather than failing a run.
#
# Corrected 2026-07-28: this pointed at ${R2_TRANSFER_GAP}, which holds only
# gpt2-large, while every by-name text checkpoint was staged under
# ${R2_GPFS_ROOT}/models beside the protein arms. The mismatch was silent by
# construction -- inspect_member is written to tolerate partial staging, so it
# recorded each missing rung as "model directory does not exist" and the run
# continued -- and it cost the H200 campaign of run 20260728164511 its entire
# text ladder: that convergence control fitted a text side of gpt2-large ALONE
# (ladder_used = gpt2-large, protgpt2, zymctrl, progen2-small, progen2-base,
# progen2-medium; gpt2, gpt2-medium, gpt2-xl and all three ByGPT5 rungs absent),
# which is precisely the n=1 text side the campaign existed to remove. The L20
# run of the same stage, where the models are all under one root, used twelve
# members and is the one its numbers came from. Tolerating partial staging is
# right; pointing at the wrong directory is not, and the two together turn a
# staging accident into a quietly wrong fit.
export R2_TEXT_MODEL_BASE_DIR="${R2_GPFS_ROOT}/models"

# --------------------------------------------------------------------- corpora
export R2_OPENWEBTEXT_DIR="${R2_TRANSFER_GAP}/openwebtext-screen/plain_text"
export R2_SWISSPROT_FASTA="${R2_REPO_ROOT}/data/swissprot/uniprot_sprot.fasta.gz"
export R2_ZYMCTRL_FASTA="${R2_REPO_ROOT}/data/zymctrl/ec_labeled_swissprot.fasta"
export R2_PFAM_RESIDUE_TSV="${R2_REPO_ROOT}/data/interpro/pfam_residue.tsv"
export R2_ALPHAFOLD_DIR="${R2_REPO_ROOT}/data/alphafold"
export R2_PROTEINGYM_DIR="${R2_REPO_ROOT}/data/proteingym/DMS_ProteinGym_substitutions"

# ----------------------------------------------------------------- interpreter
# The pod's system interpreter. There is no conda environment here, and the
# packages the measurements need (torch 2.7.1+cu128, transformers 4.52.4, numpy,
# scipy, scikit-learn, pyarrow, tokenizers) are installed against this one.
export R2_PYTHON=/opt/ac2/bin/python3

# The package root, so `import src.transfer...` resolves. Scripts also insert
# their own parents[2], but the worker may import the modules directly.
R2_PACKAGE_ROOT="${R2_PACKAGE_ROOT:-${R2_GPFS_ROOT}/biocc/Research2/transfer_package}"
export R2_PACKAGE_ROOT
export PYTHONPATH="${R2_PACKAGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Every checkpoint is on disk. Without these, an absent or mis-typed local path
# is treated by transformers as a Hub repository id and the run stalls on a
# network that is not there, instead of failing on the path that is wrong.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${R2_GPFS_ROOT}/biocc/Research2/hf_home"

# Tokenizers forks per worker process; leaving this unset prints a warning per
# fork and can deadlock when several arms run in parallel.
export TOKENIZERS_PARALLELISM=false
