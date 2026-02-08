#!/bin/bash
cd "$(dirname "$0")"
source myvenv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6

echo "=== COMET Evaluation Start: $(date) ===" 
echo "Checkpoint: best-epoch261-valloss0.021896.ckpt"
echo "Samples: 2000, Steps: 5"
echo "==========================================="

python evaluate.py \
    --config_path configs/comet_camus.yaml \
    --model_paths checkpoints/comet_camus/best-epoch261-valloss0.021896.ckpt \
    --num_samples 2000 \
    --num_inference_steps 5

echo "=== Evaluation Finished: $(date) ==="
