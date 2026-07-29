#!/bin/bash
# ============================================================================
# BioInterpretability - Data & Model Download Script
# ============================================================================
# Disk usage estimate:
#   Models:    ~25 GB (3B ~12GB, 650M ~2.5GB, InterProt SAEs ~10GB)
#   Databases: ~150 GB (UniRef50 ~25GB, Swiss-Prot ~1GB, AlphaFold human ~23GB,
#                       ClinVar ~2GB, gnomAD exome ~60GB, InterPro ~30GB, PDB ~10GB)
#   Total:     ~175 GB
#
# Features:
#   - HuggingFace models use the mirror endpoint (hf-mirror.com)
#   - Other downloads use a local proxy (127.0.0.1:54199, Xray/SSH tunnel)
#   - Fully downloaded files are skipped automatically (validated via HTTP Content-Length)
#   - Resume downloads is supported (wget -c)
#   - Safe to rerun and idempotent
# ============================================================================

set -e

# ---- Configuration ----
export HF_ENDPOINT=https://hf-mirror.com
: "${HF_TOKEN:?Set HF_TOKEN in your shell environment before running this script}"
MODEL_DIR="/Data/public"
DATA_DIR="/Data/lzp/BioInterpretebility-CC/data"

# Proxy configuration (see the Xray/SSH tunnel in ~/claude-start.sh)
PROXY_PORT="${LOCAL_PROXY_PORT:-54199}"
PROXY="http://127.0.0.1:${PROXY_PORT}"

# Xray configuration (keep in sync with ~/claude-start.sh)
VPS_IP="23.252.106.107"
XRAY_UUID="eae00b21-4f91-4fe7-98bd-0154fb6124cc"
XRAY_PUBLIC_KEY="ZFusPWpaIX-nZ5xl3hd0XhZRXFGoibqxJ2DrMr-muTY"
XRAY_SHORT_ID="068edcba2c45be5f"
XRAY_SNI="www.logitech.com"
XRAY_SERVER_PORT=443
XRAY_BIN="${HOME}/.local/bin/xray"
XRAY_CONFIG_DIR="${HOME}/.config/xray"
XRAY_LOG="${XRAY_CONFIG_DIR}/xray-download.log"
XRAY_PID_STARTED=""  # Xray PID started by this script; cleaned up on exit

mkdir -p "$DATA_DIR"/{uniref50,swissprot,interpro,clinvar,gnomad,alphafold,pdb,phosphosite,go}

# ---- Utility functions ----

# Clean up the Xray process started by this script
cleanup_xray() {
    if [ -n "$XRAY_PID_STARTED" ] && kill -0 "$XRAY_PID_STARTED" 2>/dev/null; then
        echo ""
        echo "Shutting down the Xray process started by this script (PID: $XRAY_PID_STARTED)..."
        kill "$XRAY_PID_STARTED" 2>/dev/null || true
        wait "$XRAY_PID_STARTED" 2>/dev/null || true
        echo "Xray shut down."
    fi
}
trap cleanup_xray EXIT INT TERM

# Start Xray automatically if the port is not already in use
start_xray_if_needed() {
    # Skip if a proxy is not needed
    if [ "${NO_PROXY_DOWNLOAD:-}" = "1" ]; then
        return 0
    fi

    # Check whether a proxy is already running on the port
    if curl -s --max-time 3 --proxy "$PROXY" "https://httpbin.org/ip" >/dev/null 2>&1; then
        echo "Proxy is ready (${PROXY}); reusing the existing connection."
        return 0
    fi

    # Check whether the Xray binary exists
    if [ ! -x "$XRAY_BIN" ] && ! command -v xray >/dev/null 2>&1; then
        echo "WARNING: Xray is not installed. Run claude-start first to install Xray."
        echo "  Or install it manually: https://github.com/XTLS/Xray-core/releases"
        return 1
    fi
    local xray_cmd="${XRAY_BIN}"
    if [ ! -x "$xray_cmd" ]; then
        xray_cmd="$(command -v xray)"
    fi

    echo "Proxy is not running. Starting Xray automatically (VLESS+Reality)..."

    # Generate a temporary configuration
    mkdir -p "$XRAY_CONFIG_DIR"
    local config_file="${XRAY_CONFIG_DIR}/download-client.json"
    cat > "$config_file" << XEOF
{
  "log": { "loglevel": "warning" },
  "inbounds": [{
    "tag": "http-in", "listen": "127.0.0.1", "port": ${PROXY_PORT}, "protocol": "http"
  }],
  "outbounds": [{
    "tag": "vless-reality", "protocol": "vless",
    "settings": { "vnext": [{ "address": "${VPS_IP}", "port": ${XRAY_SERVER_PORT},
      "users": [{ "id": "${XRAY_UUID}", "encryption": "none", "flow": "xtls-rprx-vision" }]
    }]},
    "streamSettings": { "network": "tcp", "security": "reality",
      "realitySettings": { "serverName": "${XRAY_SNI}", "fingerprint": "chrome",
        "publicKey": "${XRAY_PUBLIC_KEY}", "shortId": "${XRAY_SHORT_ID}" }
    }
  }]
}
XEOF

    # Start Xray
    "$xray_cmd" run -c "$config_file" > "$XRAY_LOG" 2>&1 &
    XRAY_PID_STARTED=$!

    # Wait for the port to become ready
    local retries=0
    while [ $retries -lt 20 ]; do
        if ! kill -0 "$XRAY_PID_STARTED" 2>/dev/null; then
            echo "ERROR: Xray failed to start! Log:"
            tail -5 "$XRAY_LOG" 2>/dev/null || true
            XRAY_PID_STARTED=""
            return 1
        fi
        if curl -s --max-time 3 --proxy "$PROXY" "https://httpbin.org/ip" >/dev/null 2>&1; then
            break
        fi
        sleep 1
        retries=$((retries + 1))
    done

    if [ $retries -ge 20 ]; then
        echo "ERROR: Xray startup timed out! Log:"
        tail -5 "$XRAY_LOG" 2>/dev/null || true
        kill "$XRAY_PID_STARTED" 2>/dev/null || true
        XRAY_PID_STARTED=""
        return 1
    fi

    # Show the egress IP
    local exit_ip
    exit_ip=$(curl -s --max-time 5 --proxy "$PROXY" "https://ipinfo.io/ip" 2>/dev/null || echo "unknown")
    echo "Xray started (PID: $XRAY_PID_STARTED), egress IP: ${exit_ip}"
    return 0
}

# Check whether the proxy is available
check_proxy() {
    echo -n "Checking proxy ${PROXY} ... "
    if [ "${NO_PROXY_DOWNLOAD:-}" = "1" ]; then
        echo "SKIP (NO_PROXY_DOWNLOAD=1)"
        return 0
    fi
    if curl -s --max-time 5 --proxy "$PROXY" "https://httpbin.org/ip" >/dev/null 2>&1; then
        echo "OK"
        return 0
    else
        echo "FAILED"
        echo "ERROR: Proxy connection failed. Cannot continue downloading."
        exit 1
    fi
}

# Proxy-aware wget: resume downloads + skip completed files
# Usage: dl <url> <output_dir> [expected_filename]
dl() {
    local url="$1"
    local outdir="$2"
    local filename="${3:-$(basename "$url")}"
    local filepath="${outdir}/${filename}"

    # Check whether the file has already been fully downloaded
    if [ -f "$filepath" ]; then
        # Get the local file size
        local local_size
        local_size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)

        # Get the remote file size via a HEAD request
        local remote_size
        if [ "${NO_PROXY_DOWNLOAD:-}" = "1" ]; then
            remote_size=$(curl -sI --max-time 10 "$url" 2>/dev/null | grep -i 'Content-Length' | tail -1 | tr -d '[:space:]' | cut -d: -f2)
        else
            remote_size=$(curl -sI --max-time 10 --proxy "$PROXY" "$url" 2>/dev/null | grep -i 'Content-Length' | tail -1 | tr -d '[:space:]' | cut -d: -f2)
        fi

        if [ -n "$remote_size" ] && [ "$remote_size" -gt 0 ] 2>/dev/null && [ "$local_size" -eq "$remote_size" ]; then
            echo "  SKIP (already complete): $filename ($((local_size/1024/1024)) MB)"
            return 0
        elif [ -n "$remote_size" ] && [ "$remote_size" -gt 0 ] 2>/dev/null; then
            echo "  RESUME: $filename (local: $((local_size/1024/1024)) MB, remote: $((remote_size/1024/1024)) MB)"
        else
            # Could not get the remote size (some servers do not return Content-Length); just check whether the local file is non-empty
            if [ "$local_size" -gt 1000 ]; then
                echo "  SKIP (exists, cannot verify size): $filename ($((local_size/1024/1024)) MB)"
                return 0
            fi
        fi
    fi

    # Download
    local wget_opts=(-c --tries=3 --timeout=60 --waitretry=5 -P "$outdir")
    if [ "${NO_PROXY_DOWNLOAD:-}" != "1" ]; then
        wget_opts+=(-e "use_proxy=yes" -e "https_proxy=${PROXY}" -e "http_proxy=${PROXY}")
    fi

    wget "${wget_opts[@]}" "$url" || {
        echo "  WARNING: Failed to download $filename, will retry on next run"
        return 1
    }
}

# Proxy-aware git clone: skip if the repository already exists
git_clone() {
    local url="$1"
    local dir="$2"
    local name=$(basename "$dir")

    if [ -d "$dir" ] && [ -d "$dir/.git" ]; then
        echo "  SKIP (already cloned): $name"
        return 0
    fi

    if [ "${NO_PROXY_DOWNLOAD:-}" != "1" ]; then
        git -c "http.proxy=${PROXY}" -c "https.proxy=${PROXY}" clone "$url" "$dir"
    else
        git clone "$url" "$dir"
    fi
}

# ---- Main flow ----

echo "============================================"
echo "  BioInterpretability Download Script"
echo "============================================"
echo "  MODEL_DIR: $MODEL_DIR"
echo "  DATA_DIR:  $DATA_DIR"
echo "  PROXY:     $PROXY"
echo "============================================"
echo ""

# Part 1 does not need a proxy (it uses the HF mirror); Parts 2-4 do
echo "============================================"
echo "  Part 1: Models (HuggingFace Mirror)"
echo "============================================"

# ---- 1.1 ESM-2-3B (primary model, ~12GB) ----
echo "[1/4] ESM-2-3B..."
if [ -f "$MODEL_DIR/esm2_t36_3B_UR50D/model.safetensors" ] || \
   [ -f "$MODEL_DIR/esm2_t36_3B_UR50D/pytorch_model.bin" ]; then
    echo "  SKIP (already downloaded): esm2_t36_3B_UR50D"
else
    huggingface-cli download --resume-download facebook/esm2_t36_3B_UR50D \
        --local-dir "$MODEL_DIR/esm2_t36_3B_UR50D" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN"
fi

# ---- 1.2 ESM-2-650M (comparison model, ~2.5GB) ----
echo "[2/4] ESM-2-650M..."
if [ -f "$MODEL_DIR/esm2_t33_650M_UR50D/model.safetensors" ] || \
   [ -f "$MODEL_DIR/esm2_t33_650M_UR50D/pytorch_model.bin" ]; then
    echo "  SKIP (already downloaded): esm2_t33_650M_UR50D"
else
    huggingface-cli download --resume-download facebook/esm2_t33_650M_UR50D \
        --local-dir "$MODEL_DIR/esm2_t33_650M_UR50D" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN"
fi

# ---- 1.3 InterProt pre-trained SAE (650M baseline) ----
echo "[3/4] InterProt pre-trained SAEs..."
if [ -d "$MODEL_DIR/InterProt-ESM2-SAEs" ] && \
   [ "$(ls -1 "$MODEL_DIR/InterProt-ESM2-SAEs"/*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  SKIP (already downloaded): InterProt-ESM2-SAEs"
else
    huggingface-cli download --resume-download liambai/InterProt-ESM2-SAEs \
        --local-dir "$MODEL_DIR/InterProt-ESM2-SAEs" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN"
fi

# ---- 1.4 InterPLM pre-trained SAE (650M) ----
echo "[4/4] InterPLM pre-trained SAEs..."
if [ -d "$MODEL_DIR/InterPLM-esm2-650m" ] && \
   [ "$(ls -1 "$MODEL_DIR/InterPLM-esm2-650m"/*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  SKIP (already downloaded): InterPLM-esm2-650m"
else
    huggingface-cli download --resume-download Elana/InterPLM-esm2-650m \
        --local-dir "$MODEL_DIR/InterPLM-esm2-650m" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN"
fi

echo ""
echo "============================================"
echo "  Part 2-3: Datasets (via proxy)"
echo "============================================"

# Start the proxy automatically if needed, then verify connectivity
start_xray_if_needed
check_proxy

echo ""
echo "---- Part 2: Training Data ----"

# ---- 2.1 UniRef50 (~25GB compressed) ----
echo "[UniRef50] SAE training data..."
dl "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz" \
   "$DATA_DIR/uniref50"

echo ""
echo "---- Part 3: Evaluation & Annotation Databases ----"

# ---- 3.1 Swiss-Prot (~90MB + ~800MB compressed) ----
echo "[Swiss-Prot] Annotation alignment evaluation..."
dl "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz" \
   "$DATA_DIR/swissprot"
dl "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz" \
   "$DATA_DIR/swissprot"

# ---- 3.2 InterPro ----
echo "[InterPro] Domain/motif annotations..."
dl "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/protein2ipr.dat.gz" \
   "$DATA_DIR/interpro"
dl "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list" \
   "$DATA_DIR/interpro"
dl "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/interpro.xml.gz" \
   "$DATA_DIR/interpro"

# ---- 3.3 Gene Ontology ----
echo "[GO] Gene Ontology..."
dl "http://purl.obolibrary.org/obo/go/go-basic.obo" \
   "$DATA_DIR/go"
dl "https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz" \
   "$DATA_DIR/go"

# ---- 3.4 ClinVar ----
echo "[ClinVar] Pathogenic variants..."
dl "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz" \
   "$DATA_DIR/clinvar"
dl "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi" \
   "$DATA_DIR/clinvar"
dl "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz" \
   "$DATA_DIR/clinvar"

# ---- 3.5 gnomAD v4 ----
echo "[gnomAD] Population frequencies..."
# Constraint metrics (gene-level, ~50MB, required)
dl "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv" \
   "$DATA_DIR/gnomad"

# Full exome VCFs (per chromosome, ~60GB total)
# Downloaded by default. Comment out this block if not needed.
for chr in {1..22} X Y; do
    echo "  [gnomAD] chr${chr}..."
    dl "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr${chr}.vcf.bgz" \
       "$DATA_DIR/gnomad"
    dl "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr${chr}.vcf.bgz.tbi" \
       "$DATA_DIR/gnomad"
done

# ---- 3.6 AlphaFold DB (human proteome) ----
echo "[AlphaFold] Human proteome 3D structures..."
dl "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar" \
   "$DATA_DIR/alphafold"

# ---- 3.7 BioLiP ----
echo "[BioLiP] PDB ligand-protein contacts..."
dl "https://zhanggroup.org/BioLiP/download/BioLiP_nr.txt.gz" \
   "$DATA_DIR/pdb" || echo "  WARNING: BioLiP may require manual download: https://zhanggroup.org/BioLiP/download.cgi"

# ---- 3.8 UniProt ID mapping ----
echo "[UniProt] Human ID mapping..."
dl "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz" \
   "$DATA_DIR/swissprot"

echo ""
echo "============================================"
echo "  Part 4: Code Repositories (via proxy)"
echo "============================================"

CODE_DIR="/Data/lzp/BioInterpretebility-CC/external"
mkdir -p "$CODE_DIR"

echo "[1/3] InterPLM..."
git_clone "https://github.com/ElanaPearl/InterPLM.git" "$CODE_DIR/InterPLM"

echo "[2/3] InterProt..."
git_clone "https://github.com/etowahadams/interprot.git" "$CODE_DIR/interprot"

echo "[3/3] Sparsify..."
git_clone "https://github.com/EleutherAI/sparsify.git" "$CODE_DIR/sparsify"

echo ""
echo "============================================"
echo "  Download Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Decompress UniRef50:   gunzip -k $DATA_DIR/uniref50/uniref50.fasta.gz"
echo "  2. Decompress Swiss-Prot: gunzip -k $DATA_DIR/swissprot/uniprot_sprot.xml.gz"
echo "  3. Extract AlphaFold:     tar xf $DATA_DIR/alphafold/UP000005640_9606_HUMAN_v6.tar -C $DATA_DIR/alphafold/"
echo ""
echo "  4. PhosphoSitePlus must be downloaded manually (academic license required):"
echo "     Apply at: https://www.phosphosite.org/staticDownloads"
echo "     Place the downloaded files in: $DATA_DIR/phosphosite/"
echo "     Required files:"
echo "       - Phosphorylation_site_dataset.gz"
echo "       - Ubiquitination_site_dataset.gz"
echo "       - Acetylation_site_dataset.gz"
echo "       - Methylation_site_dataset.gz"
echo "       - Regulatory_sites.gz"
echo "       - Disease-associated_sites.gz"
echo "       - Kinase_Substrate_Dataset.gz"
echo ""
echo "Storage estimate:"
echo "  Models:    ~25 GB"
echo "  Databases: ~100-200 GB (depending on the gnomAD download scope)"
echo "  Total:     ~125-225 GB"
