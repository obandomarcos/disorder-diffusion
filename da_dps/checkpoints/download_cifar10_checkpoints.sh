#!/bin/bash

################################################################################
# Download Pre-trained Checkpoints for CIFAR-10 DA-DPS
# 
# This script downloads pre-trained models from common sources
# Usage: bash download_cifar10_checkpoints.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CHECKPOINT_DIR="checkpoints/cifar10"
SCORE_NET_DIR="${CHECKPOINT_DIR}/score_network"
DPS_DIR="${CHECKPOINT_DIR}/dps_sampler"
DA_DPS_DIR="${CHECKPOINT_DIR}/da_dps_sampler"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}CIFAR-10 Checkpoint Downloader${NC}"
echo -e "${BLUE}================================${NC}"

# Create directories
echo -e "\n${YELLOW}[1/4] Creating directories...${NC}"
mkdir -p "${SCORE_NET_DIR}"
mkdir -p "${DPS_DIR}"
mkdir -p "${DA_DPS_DIR}"
echo -e "${GREEN}✓ Directories created${NC}"

# ============================================================================
# OPTION 1: Download from Hugging Face (Recommended)
# ============================================================================

download_from_huggingface() {
    echo -e "\n${YELLOW}[2/4] Downloading from Hugging Face...${NC}"
    
    # Example: Replace with actual model hub links
    HF_REPO="marcosobando/da-dps-cifar10-checkpoints"
    
    echo -e "${BLUE}Using repository: ${HF_REPO}${NC}"
    
    # Check if huggingface_hub is installed
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo -e "${YELLOW}Installing huggingface_hub...${NC}"
        pip install huggingface_hub -q
    fi
    
    # Download score network
    echo -e "${YELLOW}  Downloading score network...${NC}"
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='score_network/latest.pt',
    local_dir='${CHECKPOINT_DIR}',
    cache_dir='.cache'
)
" && echo -e "${GREEN}  ✓ Score network downloaded${NC}" || echo -e "${RED}  ✗ Failed${NC}"
    
    # Download DPS sampler
    echo -e "${YELLOW}  Downloading DPS sampler...${NC}"
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='dps_sampler/latest.pt',
    local_dir='${CHECKPOINT_DIR}',
    cache_dir='.cache'
)
" && echo -e "${GREEN}  ✓ DPS sampler downloaded${NC}" || echo -e "${RED}  ✗ Failed${NC}"
    
    # Download DA-DPS sampler
    echo -e "${YELLOW}  Downloading DA-DPS sampler...${NC}"
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='da_dps_sampler/latest.pt',
    local_dir='${CHECKPOINT_DIR}',
    cache_dir='.cache'
)
" && echo -e "${GREEN}  ✓ DA-DPS sampler downloaded${NC}" || echo -e "${RED}  ✗ Failed${NC}"
}

# ============================================================================
# OPTION 2: Download from Google Drive (if shared)
# ============================================================================

download_from_google_drive() {
    echo -e "\n${YELLOW}[2/4] Downloading from Google Drive...${NC}"
    
    # File IDs (replace with actual Google Drive file IDs)
    SCORE_NET_ID="1abc123def456"
    DPS_SAMPLER_ID="2xyz789uvw012"
    DA_DPS_SAMPLER_ID="3ghi345jkl678"
    
    # Check if gdown is installed
    if ! command -v gdown &> /dev/null; then
        echo -e "${YELLOW}Installing gdown...${NC}"
        pip install gdown -q
    fi
    
    # Download score network
    echo -e "${YELLOW}  Downloading score network...${NC}"
    gdown "${SCORE_NET_ID}" -O "${SCORE_NET_DIR}/latest.pt" && \
        echo -e "${GREEN}  ✓ Score network downloaded${NC}" || \
        echo -e "${RED}  ✗ Failed to download${NC}"
    
    # Download DPS sampler
    echo -e "${YELLOW}  Downloading DPS sampler...${NC}"
    gdown "${DPS_SAMPLER_ID}" -O "${DPS_DIR}/latest.pt" && \
        echo -e "${GREEN}  ✓ DPS sampler downloaded${NC}" || \
        echo -e "${RED}  ✗ Failed to download${NC}"
    
    # Download DA-DPS sampler
    echo -e "${YELLOW}  Downloading DA-DPS sampler...${NC}"
    gdown "${DA_DPS_SAMPLER_ID}" -O "${DA_DPS_DIR}/latest.pt" && \
        echo -e "${GREEN}  ✓ DA-DPS sampler downloaded${NC}" || \
        echo -e "${RED}  ✗ Failed to download${NC}"
}

# ============================================================================
# OPTION 3: Download from URL (direct links)
# ============================================================================

download_from_url() {
    echo -e "\n${YELLOW}[2/4] Downloading from URL...${NC}"
    
    # Replace these with actual URLs
    SCORE_NET_URL="https://example.com/checkpoints/score_network.pt"
    DPS_SAMPLER_URL="https://example.com/checkpoints/dps_sampler.pt"
    DA_DPS_SAMPLER_URL="https://example.com/checkpoints/da_dps_sampler.pt"
    
    # Download score network
    echo -e "${YELLOW}  Downloading score network...${NC}"
    if wget -q "${SCORE_NET_URL}" -O "${SCORE_NET_DIR}/latest.pt"; then
        echo -e "${GREEN}  ✓ Score network downloaded${NC}"
    else
        echo -e "${RED}  ✗ Failed to download${NC}"
    fi
    
    # Download DPS sampler
    echo -e "${YELLOW}  Downloading DPS sampler...${NC}"
    if wget -q "${DPS_SAMPLER_URL}" -O "${DPS_DIR}/latest.pt"; then
        echo -e "${GREEN}  ✓ DPS sampler downloaded${NC}"
    else
        echo -e "${RED}  ✗ Failed to download${NC}"
    fi
    
    # Download DA-DPS sampler
    echo -e "${YELLOW}  Downloading DA-DPS sampler...${NC}"
    if wget -q "${DA_DPS_SAMPLER_URL}" -O "${DA_DPS_DIR}/latest.pt"; then
        echo -e "${GREEN}  ✓ DA-DPS sampler downloaded${NC}"
    else
        echo -e "${RED}  ✗ Failed to download${NC}"
    fi
}

# ============================================================================
# OPTION 4: Train from scratch (no download)
# ============================================================================

train_from_scratch() {
    echo -e "\n${YELLOW}[2/4] Training from scratch...${NC}"
    echo -e "${BLUE}Run: python train_score_network.py${NC}"
}

# ============================================================================
# VERIFY DOWNLOADS
# ============================================================================

verify_checkpoints() {
    echo -e "\n${YELLOW}[3/4] Verifying downloads...${NC}"
    
    check_file() {
        if [ -f "$1" ]; then
            SIZE_MB=$(du -m "$1" | cut -f1)
            echo -e "${GREEN}  ✓ $1 (${SIZE_MB} MB)${NC}"
            return 0
        else
            echo -e "${RED}  ✗ $1 (missing)${NC}"
            return 1
        fi
    }
    
    check_file "${SCORE_NET_DIR}/latest.pt" || true
    check_file "${DPS_DIR}/latest.pt" || true
    check_file "${DA_DPS_DIR}/latest.pt" || true
}

# ============================================================================
# MAIN MENU
# ============================================================================

show_menu() {
    echo -e "\n${BLUE}Choose download source:${NC}"
    echo "  1) Hugging Face (recommended if available)"
    echo "  2) Google Drive (if shared)"
    echo "  3) Direct URL (custom links)"
    echo "  4) Train from scratch (no download)"
    echo "  5) Skip (verify existing)"
    echo ""
    read -p "Enter choice (1-5): " CHOICE
    
    case $CHOICE in
        1) download_from_huggingface ;;
        2) download_from_google_drive ;;
        3) download_from_url ;;
        4) train_from_scratch ;;
        5) echo -e "${YELLOW}Skipping download...${NC}" ;;
        *) echo -e "${RED}Invalid choice${NC}"; exit 1 ;;
    esac
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================

summary() {
    echo -e "\n${YELLOW}[4/4] Summary${NC}"
    echo -e "${BLUE}Checkpoint directory structure:${NC}"
    echo "  checkpoints/cifar10/"
    echo "  ├── score_network/          (pre-trained diffusion model)"
    echo "  ├── dps_sampler/            (DPS configuration)"
    echo "  └── da_dps_sampler/         (DA-DPS configuration)"
    
    if [ -f "${SCORE_NET_DIR}/latest.pt" ]; then
        echo -e "\n${GREEN}✓ Ready to use!${NC}"
        echo -e "  Run: ${BLUE}python uq_experiment_all_operators.py${NC}"
    else
        echo -e "\n${YELLOW}⚠ No checkpoints found${NC}"
        echo -e "  Train with: ${BLUE}python train_score_network.py${NC}"
    fi
}

# ============================================================================
# EXECUTE
# ============================================================================

show_menu
verify_checkpoints
summary

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}================================${NC}"
