#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H200_HELPERS="${H200_HELPERS:-${HOME}/hangzhou-remote/ssh_tunnel}"
PUSH_SCRIPT="${H200_HELPERS}/h200_push.sh"
POD_BASH="${H200_HELPERS}/h200_pod_bash.sh"
POD="${H200_POD:-${POD:-}}"
TMP_DIR="${TMP_DIR:-/tmp/biocc_h200_sync}"
KEEP_REMOTE_UPLOADS="${KEEP_REMOTE_UPLOADS:-0}"

[[ -n "$POD" ]] || {
    echo "Set H200_POD (or POD) to a running pod with GPFS and OSS mounted." >&2
    exit 2
}
[[ -x "$PUSH_SCRIPT" && -x "$POD_BASH" ]] || {
    echo "H200 helpers are missing under: $H200_HELPERS" >&2
    exit 1
}

mkdir -p "$TMP_DIR"

run_pod() {
    "$POD_BASH" "$POD" "$1"
}

extract_remote_tar() {
    local remote_tar="$1"
    local dest_dir="$2"
    run_pod "mkdir -p '$dest_dir' && tar -xf '$remote_tar' -C '$dest_dir'"
    if [[ "$KEEP_REMOTE_UPLOADS" != "1" ]]; then
        run_pod "rm -f '$remote_tar'"
    fi
}

link_file() {
    local source_path="$1"
    local link_path="$2"
    run_pod "mkdir -p '$(dirname "$link_path")' && ln -sfn '$source_path' '$link_path'"
}

prepare_tar() {
    local tar_path="$1"
    shift
    local base_dir="$1"
    shift
    echo "Creating tar: $tar_path"
    rm -f "$tar_path"
    tar -C "$base_dir" -cf "$tar_path" "$@"
}

sync_file() {
    local src="$1"
    local dest="$2"
    echo
    echo "== Sync file =="
    echo "SRC:  $src"
    echo "DEST: $dest"
    "$PUSH_SCRIPT" "$src" "$dest"
}

sync_tar() {
    local src_tar="$1"
    local remote_tar="$2"
    local extract_dir="$3"
    local top_entry
    local remote_extract_path
    local marker_path
    local source_sha
    local remote_sha
    # Avoid SIGPIPE under `set -o pipefail`: `head -n 1` can cause `tar -tf`
    # to exit non-zero after the first line has already been read.
    top_entry="$(tar -tf "$src_tar" | sed -n '1p' | cut -d/ -f1)"
    remote_extract_path="${extract_dir%/}/${top_entry}"
    marker_path="${remote_extract_path%/}/.sync_sha256"
    source_sha="$(sha256sum "$src_tar" | awk '{print $1}')"
    remote_sha="$(run_pod "if [[ -f '$marker_path' ]]; then cat '$marker_path'; else echo MISSING; fi" | tail -n 1)"
    if [[ -n "$top_entry" && "$remote_sha" == "$source_sha" ]]; then
        echo
        echo "== Skip tar =="
        echo "SRC:  $src_tar"
        echo "DEST: $remote_tar"
        echo "Reason: matching SHA-256 marker: $marker_path"
        return
    fi
    echo
    echo "== Sync tar =="
    echo "SRC:  $src_tar"
    echo "DEST: $remote_tar"
    "$PUSH_SCRIPT" "$src_tar" "$remote_tar"
    extract_remote_tar "$remote_tar" "$extract_dir"
    run_pod "mkdir -p '$remote_extract_path' && printf '%s\\n' '$source_sha' > '$marker_path'"
}

# Build compact tarballs only for directory-shaped assets.
# R0/R1 were retired to archive/retired_research_roots/ and R2's
# circuit_analysis tree was retired to archive/legacy/r2_retired_scope_20260729/
# on 2026-07-29 (docs/REPOSITORY_STRUCTURE.md); the local source paths below
# were repointed there, not touched.
prepare_tar \
    "${TMP_DIR}/r1_final_checkpoints.tar" \
    "${ROOT}/archive/retired_research_roots/r1_encoder_interpretability_benchmark/results" \
    "final_checkpoints/r1_h200_2gpu_20260401"

prepare_tar \
    "${TMP_DIR}/r1_annotation_alignment.tar" \
    "${ROOT}/archive/retired_research_roots/r1_encoder_interpretability_benchmark/results" \
    "annotation_alignment"

prepare_tar \
    "${TMP_DIR}/r2_progen2_medium_ckpt.tar" \
    "${ROOT}/results/final_checkpoints" \
    "r2_clt_progen2_medium_rerun_20260403"

prepare_tar \
    "${TMP_DIR}/r2_zymctrl_v1_ckpt.tar" \
    "${ROOT}/results/final_checkpoints" \
    "r2_clt_zymctrl_rerun_20260403"

prepare_tar \
    "${TMP_DIR}/r2_circuit_analysis_zymctrl.tar" \
    "${ROOT}/archive/legacy/r2_retired_scope_20260729/results/circuit_analysis" \
    "zymctrl"

# Core R1 data/files
sync_file \
    "${ROOT}/data/clinvar/variant_summary.txt.gz" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/clinvar/variant_summary.txt.gz"

sync_file \
    "${ROOT}/data/mechanism/gerasimavicius2022_TableS1.tsv" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/mechanism/gerasimavicius2022_TableS1.tsv"

sync_file \
    "${ROOT}/data/mechanism/badonyi2025_table_S1.csv" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/mechanism/badonyi2025_table_S1.csv"

sync_file \
    "${ROOT}/data/go/goa_uniprot_all.gaf.gz" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/go/goa_uniprot_all.gaf.gz"

sync_file \
    "${ROOT}/data/interpro/pfam_residue.tsv" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/interpro/pfam_residue.tsv"

sync_file \
    "${ROOT}/data/BioLiP/BioLiP_nr.txt" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/BioLiP/BioLiP_nr.txt"

sync_file \
    "${ROOT}/data/processed/swissprot_all_max1022.pkl" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/processed/swissprot_all_max1022.pkl"

sync_file \
    "${ROOT}/data/zymctrl/ec_labeled_swissprot.fasta" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/ec_labeled_swissprot.fasta"

sync_file \
    "${ROOT}/data/zymctrl/enzyme.dat" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/enzyme.dat"

# Directory assets
sync_tar \
    "${TMP_DIR}/r1_final_checkpoints.tar" \
    "/gpfs/jiaotongdamoxing/zhk_zip/uploads/r1_final_checkpoints.tar" \
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research1/results"

sync_tar \
    "${TMP_DIR}/r1_annotation_alignment.tar" \
    "/gpfs/jiaotongdamoxing/zhk_zip/uploads/r1_annotation_alignment.tar" \
    "/oss-pvc/zhk_zip/biocc/Research1/results"

sync_tar \
    "${TMP_DIR}/r2_progen2_medium_ckpt.tar" \
    "/gpfs/jiaotongdamoxing/zhk_zip/uploads/r2_progen2_medium_ckpt.tar" \
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/final_checkpoints"

sync_tar \
    "${TMP_DIR}/r2_zymctrl_v1_ckpt.tar" \
    "/gpfs/jiaotongdamoxing/zhk_zip/uploads/r2_zymctrl_v1_ckpt.tar" \
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/final_checkpoints"

sync_tar \
    "${TMP_DIR}/r2_circuit_analysis_zymctrl.tar" \
    "/gpfs/jiaotongdamoxing/zhk_zip/uploads/r2_circuit_analysis_zymctrl.tar" \
    "/oss-pvc/zhk_zip/biocc/Research2/results/circuit_analysis"

# Make code-relative paths resolve without duplicating large blobs.
link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/uniref50/uniref50.fasta" \
    "/oss-pvc/zhk_zip/biocc/data/uniref50/uniref50.fasta"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/uniref50/uniref50.fasta" \
    "/oss-pvc/zhk_zip/data/uniref50/uniref50.fasta"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/proteingym/DMS_ProteinGym_substitutions" \
    "/oss-pvc/zhk_zip/biocc/data/proteingym/DMS_ProteinGym_substitutions"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/proteingym/DMS_ProteinGym_substitutions" \
    "/oss-pvc/zhk_zip/data/proteingym/DMS_ProteinGym_substitutions"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/clinvar/variant_summary.txt.gz" \
    "/oss-pvc/zhk_zip/biocc/data/clinvar/variant_summary.txt.gz"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/clinvar/variant_summary.txt.gz" \
    "/oss-pvc/zhk_zip/data/clinvar/variant_summary.txt.gz"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/mechanism/gerasimavicius2022_TableS1.tsv" \
    "/oss-pvc/zhk_zip/biocc/data/mechanism/gerasimavicius2022_TableS1.tsv"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/mechanism/badonyi2025_table_S1.csv" \
    "/oss-pvc/zhk_zip/biocc/data/mechanism/badonyi2025_table_S1.csv"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/mechanism/gerasimavicius2022_TableS1.tsv" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/variant_effect/gerasimavicius2022_TableS1.tsv"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/mechanism/badonyi2025_table_S1.csv" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/variant_effect/badonyi2025_table_S1.csv"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/mechanism/gerasimavicius2022_TableS1.tsv" \
    "/oss-pvc/zhk_zip/data/variant_effect/gerasimavicius2022_TableS1.tsv"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/mechanism/badonyi2025_table_S1.csv" \
    "/oss-pvc/zhk_zip/data/variant_effect/badonyi2025_table_S1.csv"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/go/goa_uniprot_all.gaf.gz" \
    "/oss-pvc/zhk_zip/biocc/data/go/goa_uniprot_all.gaf.gz"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/go/goa_uniprot_all.gaf.gz" \
    "/oss-pvc/zhk_zip/data/go/goa_uniprot_all.gaf.gz"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/go/goa_uniprot_all.gaf.gz" \
    "/oss-pvc/zhk_zip/biocc/data/go/goa_human.gaf.gz"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/interpro/pfam_residue.tsv" \
    "/oss-pvc/zhk_zip/biocc/data/interpro/pfam_residue.tsv"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/interpro/pfam_residue.tsv" \
    "/oss-pvc/zhk_zip/data/interpro/pfam_residue.tsv"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/BioLiP/BioLiP_nr.txt" \
    "/oss-pvc/zhk_zip/biocc/data/BioLiP/BioLiP_nr.txt"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/BioLiP/BioLiP_nr.txt" \
    "/oss-pvc/zhk_zip/data/BioLiP/BioLiP_nr.txt"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/BioLiP/BioLiP_nr.txt" \
    "/oss-pvc/zhk_zip/biocc/data/BioLiP/BioLiP.txt"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/processed/swissprot_all_max1022.pkl" \
    "/oss-pvc/zhk_zip/biocc/data/processed/swissprot_all_max1022.pkl"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/processed/swissprot_all_max1022.pkl" \
    "/oss-pvc/zhk_zip/data/processed/swissprot_all_max1022.pkl"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/ec_labeled_swissprot.fasta" \
    "/oss-pvc/zhk_zip/biocc/data/zymctrl/ec_labeled.fasta"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/ec_labeled_swissprot.fasta" \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/ec_labeled.fasta"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/ec_labeled_swissprot.fasta" \
    "/oss-pvc/zhk_zip/data/zymctrl/ec_labeled.fasta"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/enzyme.dat" \
    "/oss-pvc/zhk_zip/biocc/data/zymctrl/enzyme.dat"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/data/zymctrl/enzyme.dat" \
    "/oss-pvc/zhk_zip/data/zymctrl/enzyme.dat"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research1/results/final_checkpoints" \
    "/oss-pvc/zhk_zip/biocc/Research1/results/final_checkpoints"

link_file \
    "/oss-pvc/zhk_zip/biocc/Research1/results/annotation_alignment" \
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research1/results/annotation_alignment"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403" \
    "/oss-pvc/zhk_zip/biocc/Research2/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403"

link_file \
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/final_checkpoints/r2_clt_zymctrl_rerun_20260403" \
    "/oss-pvc/zhk_zip/biocc/Research2/results/final_checkpoints/r2_clt_zymctrl_rerun_20260403"

echo
echo "Required local assets have been synced to H200."
