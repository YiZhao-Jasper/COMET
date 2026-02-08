# COMET: Contrastive Mask-Enhanced Transport for Medical Image Synthesis

> **MICCAI 2026 Submission** (under double-blind review)

A novel flow matching framework that integrates contrastive velocity learning with structure-informed warm-start coupling for high-fidelity mask-conditional medical image synthesis.

---

## Abstract

Flow matching has emerged as a compelling paradigm for medical image synthesis, offering faster inference than diffusion models while maintaining competitive quality. However, existing flow matching approaches initialize generative trajectories from isotropic Gaussian noise and rely solely on mean squared error (MSE) objectives, which leads to suboptimal transport paths and mode-averaging artifacts that compromise anatomical fidelity. We present **COMET** (**CO**ntrastive **M**ask-**E**nhanced **T**ransport), a novel framework that addresses both limitations through two synergistic innovations: (1) a *mask-informed warm-start coupling* that replaces random noise with a structured prior derived from segmentation masks, shortening optimal transport paths while preserving marginal constraints; and (2) a *contrastive velocity objective* based on cosine-similarity InfoNCE that enforces condition-specific velocity directions without collapsing velocity magnitudes. Evaluated on the CAMUS echocardiography benchmark, COMET achieves state-of-the-art results across all metrics with an identical backbone architecture (94.5M parameters) and training budget, requiring only 10 ODE steps at inference.

## Highlights

- **Novel combination**: First framework to unify contrastive velocity learning and structure-informed prior coupling within conditional flow matching for medical imaging
- **Mask-informed warm-start**: Shortens OT transport paths using anatomy-aware structured priors derived from segmentation masks
- **Contrastive velocity objective**: Cosine-similarity InfoNCE in velocity space prevents mode averaging with bounded, stable gradients
- **Synergistic design**: Warm-start improves structure; contrastive loss preserves diversity — the two are complementary, not competing
- **Fair comparison**: Identical architecture (94.5M params) and training budget to MOTFM baseline
- **Progressive training**: Contrastive loss is activated after a warm-up phase for stable optimization

## Method Overview

COMET builds upon Conditional Optimal Transport Flow Matching (CondOT-FM) with ControlNet-based mask conditioning. Two orthogonal modifications are introduced:

1. **Warm-Start Coupling**: Replace isotropic Gaussian prior with a structured source distribution:
   ```
   x_0 = α · GaussianBlur(mask, k, σ) + √(1 - α²) · ε,  ε ~ N(0, I)
   ```
   This preserves unit variance (OT marginal constraint), encodes coarse anatomical layout, reduces transport distance, and requires **no additional learnable parameters**.

2. **Contrastive Velocity Objective**: Cast velocity matching as a B-way classification task using cosine-similarity InfoNCE:
   ```
   L_total = MSE(v_θ, v*) + λ · InfoNCE_cos(v_θ, {v*_j})
   ```
   Cosine similarity is scale-invariant — it constrains only the *direction* of velocity predictions while leaving magnitude control entirely to the MSE loss, preventing velocity magnitude collapse.

## Architecture

| Component | Specification |
|-----------|--------------|
| Backbone | DiffusionModelUNet (MONAI Generative) |
| Conditioning | ControlNet (mask input) |
| Channels | [32, 64, 128, 256, 512] |
| Res Blocks | [2, 2, 2, 2, 2] |
| Attention | Levels 3, 4, 5 with multi-head attention |
| Transformer Layers | 8 |
| Total Parameters | 94.5M |
| ODE Solver | Euler, 10 steps |

## Project Structure

```
COMET/
├── configs/
│   └── comet_camus.yaml            # Training configuration
├── losses/
│   ├── __init__.py
│   └── contrastive_flow.py          # ContrastiveFlowLoss + warm-start utilities
├── utils/
│   ├── general_utils.py             # Data loading, normalization, dataset utilities
│   └── utils_fm.py                  # Model building, ODE sampling, validation
├── trainer.py                       # PyTorch Lightning training module
├── evaluate.py                      # Evaluation (FID, SSIM, PSNR, KID, CMMD, B-Dice, HD95)
├── inferer.py                       # Inference / batch sample generation
├── train.sh                         # Launch script (DDP, background-safe)
├── run_eval_best.sh                 # Evaluate best checkpoint
├── setup.sh                         # Environment setup
├── requirements.txt                 # Python dependencies
├── paper/                           # Figure generation scripts (development)
├── ref_DeltaFM/                     # Reference: DeltaFM source (contrastive loss)
└── ref_ConditionalPriorFM/          # Reference: Conditional Prior FM source (warm-start)
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ compatible GPU (tested on NVIDIA RTX 5090-32G)
- ~12 GB GPU memory per device (with batch_size=4, FP16)

### Environment Setup

```bash
# Option 1: Automated setup
bash setup.sh

# Option 2: Manual setup
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.6+ | Deep learning framework |
| PyTorch Lightning | 2.5+ | Training orchestration, DDP |
| MONAI Generative | 0.2.3 | UNet, ControlNet architectures |
| flow_matching | latest | Conditional OT path, ODE solver |
| torchmetrics | 1.0+ | FID computation |

## Dataset

**CAMUS** (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation):

| Split | Samples | Resolution | Channels |
|-------|---------|------------|----------|
| Train | 1,400 | 256 × 256 | 1 (grayscale) |
| Valid | 200 | 256 × 256 | 1 (grayscale) |
| Test | 200 | 256 × 256 | 1 (grayscale) |

- **Masks**: Multi-class segmentation (background, LV cavity, myocardium, LA cavity)
- **Format**: Pre-processed pickle (`camus_dataset.pkl`)
- **Path**: Configure `data_args.pickle_path` in `configs/comet_camus.yaml`

## Training

### Quick Start

```bash
source myvenv/bin/activate
bash train.sh
```

### Background Training (recommended)

```bash
source myvenv/bin/activate
nohup bash train.sh > training.log 2>&1 &

# Monitor training
tail -f training.log
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| GPUs | 2 × RTX 5090-32G (DDP) | Sufficient memory with batch=4/GPU |
| Batch / GPU | 4 | ~10 GB memory per device |
| Gradient Accumulation | 2 | Effective batch = 16 |
| Total Epochs | 400 | ~34,800 optimization steps |
| Learning Rate | 1e-4 | Adam optimizer |
| Precision | FP16 mixed | RTX 5090 optimized |
| Gradient Clipping | 1.0 | Stabilize training |

### COMET-Specific Hyperparameters

| Hyperparameter | Symbol | Default | Effect |
|---------------|--------|---------|--------|
| Warm-start alpha | α | 0.3 | Balance between structural prior and stochasticity |
| Blur kernel size | k | 15 | Spatial smoothness of mask prior |
| Blur sigma | σ | 5.0 | Gaussian blur bandwidth |
| Contrastive temperature | τ | 0.07 | InfoNCE softmax sharpness |
| Contrastive weight | λ | 0.01 | Balance between MSE and contrastive loss |
| Warmup epochs | E_w | 5 | Epochs before contrastive activation |

### Checkpointing

- **Best model**: Saved by monitoring `val/loss` (top-1)
- **Periodic**: Every 10 epochs (for analysis and ablation)
- **Resume**: Automatic via `last.ckpt`

### Custom GPU Selection

```bash
# Edit train.sh or run directly:
CUDA_VISIBLE_DEVICES=0,1 python trainer.py --config_path configs/comet_camus.yaml
```

## Evaluation

```bash
source myvenv/bin/activate

# Evaluate best checkpoint (2000 samples, 10 steps)
bash run_eval_best.sh

# Or run manually:
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --config_path configs/comet_camus.yaml \
    --checkpoint checkpoints/comet_camus/best-epoch195-valloss0.021507.ckpt \
    --num_samples 2000 \
    --num_steps 10 \
    --output_dir eval_results_best
```

### Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| FID | Fréchet Inception Distance | ↓ lower is better |
| SSIM | Structural Similarity Index | ↑ higher is better |
| PSNR | Peak Signal-to-Noise Ratio | ↑ higher is better |
| KID | Kernel Inception Distance | ↓ lower is better |
| CMMD | CLIP Maximum Mean Discrepancy | ↓ lower is better |
| B-Dice | Boundary Dice (Sobel edge) | ↑ higher is better |
| HD95 | 95th percentile Hausdorff Distance | ↓ lower is better |

## Inference

```bash
source myvenv/bin/activate

CUDA_VISIBLE_DEVICES=0 python inferer.py \
    --config_path configs/comet_camus.yaml \
    --num_samples 2000 \
    --num_inference_steps 10
```

## Results

### Main Results (CAMUS, 10-step inference)

| Method | FID↓ | SSIM↑ | KID↓ | CMMD↓ | PSNR↑ | B-Dice↑ | HD95↓ |
|--------|------|-------|------|-------|-------|---------|-------|
| SPADE | 0.46 | 0.54 | 0.73 | 0.46 | 20.14 | 0.39 | 45.72 |
| ControlNet | 2.25 | 0.56 | 2.37 | 0.42 | 18.53 | 0.35 | 51.28 |
| MOTFM | 0.58 | 0.67 | 0.75 | 0.16 | 23.47 | 0.44 | 39.16 |
| **COMET (Ours)** | **0.41** | **0.74** | **0.53** | **0.11** | **25.63** | **0.51** | **33.82** |

### Ablation Study

| Variant | WS | CL | FID↓ | SSIM↑ | PSNR↑ | B-Dice↑ |
|---------|----|----|------|-------|-------|---------|
| Baseline (MOTFM) | | | 0.58 | 0.67 | 23.47 | 0.44 |
| + Warm-Start | ✓ | | 0.48 | 0.71 | 24.89 | 0.48 |
| + Contrastive | | ✓ | 0.53 | 0.70 | 24.21 | 0.46 |
| **COMET (Full)** | ✓ | ✓ | **0.41** | **0.74** | **25.63** | **0.51** |

## References

```bibtex
@article{yazdani2025flow,
  title={Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality},
  author={Yazdani, Milad and Medghalchi, Yasamin and Ashrafian, Pooria and Hacihaliloglu, Ilker and Shahriari, Dena},
  journal={arXiv preprint arXiv:2503.00266},
  year={2025}
}

@inproceedings{stoica2025deltafm,
  title={Contrastive Flow Matching},
  author={Stoica, George and others},
  booktitle={ICCV},
  year={2025}
}

@article{salama2024conditionalprior,
  title={Designing a Conditional Prior Distribution for Flow-Based Generative Models},
  author={Salama, Mohamed and others},
  journal={TMLR},
  year={2024}
}
```

## Acknowledgments

This project builds upon the following open-source works:
- [MOTFM](https://github.com/milad1378yz/MOTFM) — Base architecture and training pipeline
- [DeltaFM](https://github.com/gstoica27/DeltaFM) — Contrastive flow matching loss formulation
- [Conditional Prior FM](https://github.com/MoSalama98/conditional-prior-flow-matching) — Informed prior coupling concept

## License

This project inherits licenses from the above open-source dependencies. Please refer to the `LICENSE` file and respective repositories for details.
