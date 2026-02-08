#!/bin/bash
# =============================================================================
# COMET Environment Setup
# =============================================================================
# Creates a Python venv and installs all dependencies.
# Based on MOTFM's requirements + pytorch-lightning for DDP.
# =============================================================================

set -e

echo "=============================================="
echo "COMET Environment Setup"
echo "=============================================="

cd "$(dirname "$0")"

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv myvenv

# Activate
source myvenv/bin/activate

# Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
echo "[3/4] Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
echo "[4/4] Installing project dependencies..."
pip install \
    flow_matching \
    "monai-generative==0.2.3" \
    pytorch-lightning \
    torchmetrics \
    "PyYAML>=6.0" \
    numpy \
    scipy \
    matplotlib \
    tqdm \
    pytorch-fid

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate:  source myvenv/bin/activate"
echo "To train:     bash train.sh"
