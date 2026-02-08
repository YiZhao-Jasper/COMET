#!/bin/bash
# =============================================================================
# COMET Training Script — 2x RTX 5090-32G DDP
# =============================================================================
# Usage:
#   bash train.sh          # foreground
#   nohup bash train.sh > training.log 2>&1 &   # background
# =============================================================================

set -e

# Use GPUs 6,7
export CUDA_VISIBLE_DEVICES=6,7

# Avoid OOM from memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Working directory
cd "$(dirname "$0")"

echo "=============================================="
echo "COMET Training — Contrastive Mask-Enhanced Transport"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="

python trainer.py --config_path configs/comet_camus.yaml

echo "=============================================="
echo "Training finished: $(date)"
echo "=============================================="
