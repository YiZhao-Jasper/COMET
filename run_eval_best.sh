#!/bin/bash
cd "$(dirname "$0")"
source myvenv/bin/activate
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONUNBUFFERED=1

python -u evaluate.py \
    --checkpoint checkpoints/comet_camus/best-epoch195-valloss0.021507.ckpt \
    --num_samples 2000 \
    --num_steps 10 \
    --output_dir eval_results_best
