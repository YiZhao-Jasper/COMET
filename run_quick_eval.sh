#!/bin/bash
cd "$(dirname "$0")"
source myvenv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=7

BEST_CKPT=$(ls checkpoints/comet_camus/best-*.ckpt 2>/dev/null | head -1)
echo "=== Quick COMET Eval: $(date) ==="
echo "Checkpoint: $BEST_CKPT"
echo "Samples: 200 (quick test), Steps: 10"

python evaluate.py \
    --config_path configs/comet_camus.yaml \
    --model_paths "$BEST_CKPT" \
    --num_samples 200 \
    --num_inference_steps 10

echo "=== Done: $(date) ==="
