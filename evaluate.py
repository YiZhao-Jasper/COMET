"""
COMET Complete Evaluation Script (MOTFM-aligned)

Computes:
- FID (Frechet Inception Distance) - matching MOTFM's core metric
- SSIM (Structural Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- Boundary metrics: Boundary Dice, HD95, ASSD (image-based)
- Topology Error Rate
- Visualization comparison

Usage:
    CUDA_VISIBLE_DEVICES=6 python evaluate.py \
        --config_path configs/comet_camus.yaml \
        --model_paths checkpoints/comet_camus/best-epoch261-valloss0.021896.ckpt \
        --num_samples 2000 \
        --num_inference_steps 5
"""

import argparse
import json
import os
import sys
import warnings
import pickle
import time
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scipy import ndimage
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.general_utils import load_config
from utils.utils_fm import sample_batch
from trainer import FlowMatchingDataModule, FlowMatchingLightningModule


###############################################################################
# FID Computation (using Inception V3, matching MOTFM/pytorch-fid)
###############################################################################

class InceptionV3Features(nn.Module):
    """Extract features from InceptionV3 for FID computation."""
    def __init__(self, device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove final classifier
        self.model.eval()
        self.model.to(device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, x):
        # x: [B, 1, H, W] grayscale -> [B, 3, 299, 299] for Inception
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize to Inception expected range
        x = (x - 0.5) / 0.5  # [-1, 1]
        features = self.model(x)
        return features


def compute_fid_from_features(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    """
    Compute FID between two sets of Inception features.
    FID = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2*sqrt(C_r * C_g))
    """
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(gen_features, axis=0)
    
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)
    
    diff = mu_r - mu_g
    
    # Product of covariances
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    
    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2 * np.trace(covmean)
    return float(fid)


def extract_inception_features(images: List[np.ndarray], inception: InceptionV3Features, 
                                batch_size: int = 32) -> np.ndarray:
    """Extract Inception features from a list of images."""
    all_features = []
    device = inception.device
    
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        # Stack and convert to tensor [B, 1, H, W]
        batch_tensor = torch.stack([
            torch.from_numpy(img).float().unsqueeze(0) if img.ndim == 2 
            else torch.from_numpy(img).float()
            for img in batch_imgs
        ]).to(device)
        
        # Ensure values in [0, 1]
        batch_tensor = batch_tensor.clamp(0, 1)
        
        features = inception(batch_tensor).cpu().numpy()
        all_features.append(features)
    
    return np.concatenate(all_features, axis=0)


###############################################################################
# Boundary Metrics (image-based - correct approach)
###############################################################################

def extract_boundary_from_image(image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Extract boundaries from image using Sobel edge detection.
    This is the correct approach for evaluating boundary quality
    from generated images (not from condition masks).
    """
    from scipy.ndimage import sobel
    # Compute gradient magnitude
    sx = sobel(image, axis=0)
    sy = sobel(image, axis=1)
    edge_mag = np.sqrt(sx**2 + sy**2)
    # Normalize
    if edge_mag.max() > 0:
        edge_mag = edge_mag / edge_mag.max()
    return (edge_mag > threshold).astype(np.uint8)


def compute_boundary_dice(pred_boundary: np.ndarray, gt_boundary: np.ndarray) -> float:
    """Compute Dice score between two boundary maps."""
    intersection = np.sum(pred_boundary * gt_boundary)
    union = np.sum(pred_boundary) + np.sum(gt_boundary)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def compute_hd95(pred_boundary: np.ndarray, gt_boundary: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff Distance between boundaries."""
    pred_points = np.argwhere(pred_boundary > 0)
    gt_points = np.argwhere(gt_boundary > 0)
    
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 256.0  # Max possible distance for 256x256 image
    
    from scipy.spatial import distance
    d_pred_to_gt = distance.cdist(pred_points, gt_points, 'euclidean').min(axis=1)
    d_gt_to_pred = distance.cdist(gt_points, pred_points, 'euclidean').min(axis=1)
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def compute_assd(pred_boundary: np.ndarray, gt_boundary: np.ndarray) -> float:
    """Compute Average Symmetric Surface Distance."""
    pred_points = np.argwhere(pred_boundary > 0)
    gt_points = np.argwhere(gt_boundary > 0)
    
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 256.0
    
    from scipy.spatial import distance
    d_pred_to_gt = distance.cdist(pred_points, gt_points, 'euclidean').min(axis=1)
    d_gt_to_pred = distance.cdist(gt_points, pred_points, 'euclidean').min(axis=1)
    return float((d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2)


def compute_mask_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute standard Dice score on masks."""
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def compute_topology_error(pred_img: np.ndarray, gt_img: np.ndarray, threshold: float = 0.5) -> bool:
    """Check if images have different topology based on thresholded regions."""
    pred_binary = (pred_img > threshold).astype(np.uint8)
    gt_binary = (gt_img > threshold).astype(np.uint8)
    _, n_pred = ndimage.label(pred_binary)
    _, n_gt = ndimage.label(gt_binary)
    return n_pred != n_gt


###############################################################################
# Visualization
###############################################################################

def create_comparison_visualization(
    real_images: List[np.ndarray],
    gen_images: List[np.ndarray],
    masks: List[np.ndarray],
    output_dir: str,
    num_show: int = 16,
    checkpoint_name: str = "",
):
    """Create visual comparison grid between real and generated images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    num_show = min(num_show, len(real_images))
    
    # 1. Grid comparison: Real vs Generated
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    fig.suptitle(f'COMET Generated vs Real ({checkpoint_name})', fontsize=16, fontweight='bold')
    
    for i in range(min(16, num_show)):
        row = i // 4
        col_real = (i % 4) * 2
        col_gen = col_real + 1
        
        axes[row, col_real].imshow(real_images[i], cmap='gray', vmin=0, vmax=1)
        axes[row, col_real].set_title(f'Real {i+1}', fontsize=8)
        axes[row, col_real].axis('off')
        
        axes[row, col_gen].imshow(gen_images[i], cmap='gray', vmin=0, vmax=1)
        axes[row, col_gen].set_title(f'Gen {i+1}', fontsize=8)
        axes[row, col_gen].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_grid_{checkpoint_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Boundary comparison
    fig, axes = plt.subplots(3, 8, figsize=(24, 9))
    fig.suptitle(f'Boundary Quality Comparison ({checkpoint_name})', fontsize=16, fontweight='bold')
    
    for i in range(min(8, num_show)):
        real_boundary = extract_boundary_from_image(real_images[i])
        gen_boundary = extract_boundary_from_image(gen_images[i])
        
        axes[0, i].imshow(real_images[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Real {i+1}', fontsize=8)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(gen_images[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Gen {i+1}', fontsize=8)
        axes[1, i].axis('off')
        
        # Overlay boundaries
        overlay = np.zeros((*real_images[i].shape, 3))
        overlay[:, :, 0] = real_boundary.astype(float)  # Red = real boundary
        overlay[:, :, 1] = gen_boundary.astype(float)    # Green = gen boundary
        axes[2, i].imshow(overlay)
        axes[2, i].set_title(f'Boundary (R/G)', fontsize=8)
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Real', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Generated', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Boundaries', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boundary_comparison_{checkpoint_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Pixel intensity distribution (KDE) - matching MOTFM Fig.3
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    real_all = np.concatenate([img.flatten() for img in real_images[:200]])
    gen_all = np.concatenate([img.flatten() for img in gen_images[:200]])
    
    ax.hist(real_all, bins=100, alpha=0.5, label='Real', density=True, color='blue')
    ax.hist(gen_all, bins=100, alpha=0.5, label=f'COMET ({checkpoint_name})', density=True, color='red')
    ax.set_xlabel('Pixel Intensity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Pixel Intensity Distribution (cf. MOTFM Fig.3)', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pixel_distribution_{checkpoint_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualizations saved to: {output_dir}")


###############################################################################
# Main Evaluation Pipeline
###############################################################################

def evaluate_checkpoint(
    config_path: str,
    model_path: str,
    num_samples: int,
    num_inference_steps: int,
    output_dir: str,
    device: torch.device,
):
    """Run complete evaluation for a single checkpoint."""
    
    config = load_config(config_path)
    ckpt_name = os.path.basename(model_path).replace('.ckpt', '')
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {ckpt_name}")
    print(f"{'='*70}")
    
    # --- Load model ---
    print(f"  Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    lightning_module = FlowMatchingLightningModule(config)
    lightning_module.load_state_dict(checkpoint["state_dict"], strict=True)
    model = lightning_module.model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    print(f"  Model loaded (epoch={epoch}, params={sum(p.numel() for p in model.parameters()):,})")
    
    # --- Setup data ---
    datamodule = FlowMatchingDataModule(config)
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()
    dataset_size = len(val_loader.dataset)
    
    model_args = config.get("model_args", {})
    mask_cond = bool(model_args.get("mask_conditioning", False))
    class_cond = bool(model_args.get("with_conditioning", False))
    
    # [COMET] Warm-start config for inference
    comet_args = config.get("comet_args", config.get("cwsfm_args", {}))
    warm_start_config = comet_args if bool(comet_args.get("use_warm_start", False)) else None
    if warm_start_config:
        print(f"  [COMET] Warm-start enabled: alpha={comet_args.get('warm_start_alpha', 0.3)}")
    
    # --- Solver config ---
    solver_config = dict(config.get("solver_args", {}))
    solver_config["time_points"] = num_inference_steps
    solver_config["step_size"] = 1.0 / num_inference_steps
    
    print(f"  Generating {num_samples} samples ({num_inference_steps} steps)...")
    if num_samples > dataset_size:
        print(f"  Note: Reusing {dataset_size} conditions to generate {num_samples} samples")
    
    # --- Generate samples and collect real/gen pairs ---
    real_images = []
    gen_images = []
    masks_list = []
    
    samples_collected = 0
    val_iterator = iter(val_loader)
    total_gen_time = 0.0
    
    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="  Generating")
        while samples_collected < num_samples:
            try:
                batch = next(val_iterator)
            except StopIteration:
                val_iterator = iter(val_loader)
                batch = next(val_iterator)
            
            t0 = time.time()
            gen_imgs = sample_batch(
                model=model,
                solver_config=solver_config,
                batch=batch,
                device=device,
                class_conditioning=class_cond,
                mask_conditioning=mask_cond,
                warm_start_config=warm_start_config,
            )
            total_gen_time += time.time() - t0
            
            real_imgs = batch["images"]
            batch_masks = batch.get("masks", None)
            
            bs = gen_imgs.shape[0]
            for i in range(bs):
                if samples_collected >= num_samples:
                    break
                
                g = gen_imgs[i, 0].cpu().numpy().astype(np.float32)
                r = real_imgs[i, 0].cpu().numpy().astype(np.float32) if real_imgs is not None else np.zeros_like(g)
                
                # Per-sample normalize to [0, 1] (matches MOTFM's normalize_zero_to_one)
                # Flow matching models naturally output values slightly outside [0,1].
                # This is the standard post-processing step used by MOTFM.
                g_min, g_max = g.min(), g.max()
                if g_max - g_min > 1e-6:
                    g = (g - g_min) / (g_max - g_min)
                else:
                    g = np.zeros_like(g)
                r = np.clip(r, 0, 1)
                
                gen_images.append(g)
                real_images.append(r)
                
                if batch_masks is not None:
                    masks_list.append(batch_masks[i, 0].cpu().numpy())
                
                samples_collected += 1
                pbar.update(1)
        pbar.close()
    
    avg_time = total_gen_time / samples_collected
    print(f"  Generation done. Avg time/sample: {avg_time:.4f}s")
    
    # --- Compute FID ---
    print(f"  Computing FID (Inception V3 features)...")
    try:
        inception = InceptionV3Features(device)
        real_feats = extract_inception_features(real_images, inception, batch_size=32)
        gen_feats = extract_inception_features(gen_images, inception, batch_size=32)
        fid_score = compute_fid_from_features(real_feats, gen_feats)
        print(f"  FID = {fid_score:.4f}")
        del inception
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FID computation failed: {e}")
        fid_score = float('nan')
    
    # --- Compute per-sample metrics ---
    print(f"  Computing per-sample metrics (SSIM, PSNR, Boundary)...")
    all_metrics = defaultdict(list)
    
    for idx in tqdm(range(len(gen_images)), desc="  Metrics"):
        g = gen_images[idx]
        r = real_images[idx]
        
        # SSIM
        data_range = max(r.max() - r.min(), 1e-8)
        s = ssim_func(r, g, data_range=data_range)
        all_metrics['ssim'].append(s)
        
        # PSNR
        p = psnr_func(r, g, data_range=data_range)
        all_metrics['psnr'].append(p)
        
        # Boundary metrics (image-based: extract edges from images, not masks)
        gen_boundary = extract_boundary_from_image(g)
        real_boundary = extract_boundary_from_image(r)
        
        all_metrics['boundary_dice'].append(compute_boundary_dice(gen_boundary, real_boundary))
        all_metrics['boundary_hd95'].append(compute_hd95(gen_boundary, real_boundary))
        all_metrics['boundary_assd'].append(compute_assd(gen_boundary, real_boundary))
        
        # Topology
        all_metrics['topology_error'].append(float(compute_topology_error(g, r)))
        
        # Mask-based metrics (if masks available - Dice on condition mask vs threshold)
        if idx < len(masks_list):
            m = masks_list[idx]
            # Threshold generated image to get pseudo-mask and compare with condition mask
            gen_pseudo_mask = (g > 0.5).astype(np.uint8)
            gt_mask_binary = (m > 0).astype(np.uint8)
            if gt_mask_binary.sum() > 0:
                all_metrics['mask_dice'].append(compute_mask_dice(gen_pseudo_mask, gt_mask_binary))
    
    # --- Aggregate results ---
    results = {
        'checkpoint': ckpt_name,
        'epoch': epoch,
        'num_samples': num_samples,
        'num_inference_steps': num_inference_steps,
        'avg_time_per_sample': avg_time,
        'fid': fid_score,
        'metrics': {},
    }
    
    print(f"\n{'='*70}")
    print(f"Results: {ckpt_name} ({num_samples} samples, {num_inference_steps} steps)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")
    
    # FID first
    print(f"{'FID':<25} {fid_score:>10.4f} {'---':>10} {'---':>10} {'---':>10}")
    results['metrics']['fid'] = {'mean': fid_score}
    
    for name, values in sorted(all_metrics.items()):
        v = np.array(values)
        v_valid = v[np.isfinite(v)]
        if len(v_valid) > 0:
            mean, std = v_valid.mean(), v_valid.std()
            vmin, vmax = v_valid.min(), v_valid.max()
        else:
            mean = std = vmin = vmax = float('nan')
        
        print(f"{name:<25} {mean:>10.4f} {std:>10.4f} {vmin:>10.4f} {vmax:>10.4f}")
        results['metrics'][name] = {'mean': float(mean), 'std': float(std), 'min': float(vmin), 'max': float(vmax)}
    
    topo_rate = np.mean(all_metrics.get('topology_error', [0])) * 100
    print(f"\nTopology Error Rate: {topo_rate:.2f}%")
    print(f"Avg inference time: {avg_time:.4f}s/sample")
    print(f"{'='*70}")
    
    results['topology_error_rate'] = topo_rate
    
    # --- Save results ---
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f'eval_{ckpt_name}_{num_samples}samples_{num_inference_steps}steps.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")
    
    # --- Visualization ---
    print(f"  Creating visualizations...")
    create_comparison_visualization(
        real_images, gen_images, masks_list, output_dir,
        num_show=16, checkpoint_name=ckpt_name,
    )
    
    # Cleanup
    del model, gen_images, real_images
    torch.cuda.empty_cache()
    
    return results


def print_comparison_table(all_results: List[Dict]):
    """Print comparison table across checkpoints."""
    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE (COMET vs MOTFM Reference)")
    print(f"{'='*80}")
    
    # MOTFM reference values from Table 1 (Mask-conditional, M-M)
    motfm_ref = {
        '1-step':  {'fid': 3.91, 'ssim': 0.72},
        '10-step': {'fid': 0.58, 'ssim': 0.67},
        '50-step': {'fid': 0.22, 'ssim': 0.66},
    }
    
    header = f"{'Checkpoint':<35} {'FID':>8} {'SSIM':>8} {'PSNR':>8} {'BndDice':>8} {'HD95':>8} {'ASSD':>8} {'TopoErr':>8}"
    print(header)
    print("-" * 80)
    
    # Print MOTFM reference
    for step_key, vals in motfm_ref.items():
        print(f"{'MOTFM (' + step_key + ')':<35} {vals['fid']:>8.2f} {vals['ssim']:>8.4f} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    
    print("-" * 80)
    
    # Print COMET results
    for res in all_results:
        m = res['metrics']
        name = res['checkpoint'][:33]
        fid = res['fid']
        ssim_val = m.get('ssim', {}).get('mean', float('nan'))
        psnr_val = m.get('psnr', {}).get('mean', float('nan'))
        bdice = m.get('boundary_dice', {}).get('mean', float('nan'))
        hd95 = m.get('boundary_hd95', {}).get('mean', float('nan'))
        assd = m.get('boundary_assd', {}).get('mean', float('nan'))
        topo = res.get('topology_error_rate', float('nan'))
        
        print(f"{name:<35} {fid:>8.2f} {ssim_val:>8.4f} {psnr_val:>8.2f} {bdice:>8.4f} {hd95:>8.2f} {assd:>8.2f} {topo:>7.1f}%")
    
    print("=" * 80)
    print("\nMOTFM reference: Table 1, Mask-conditional (M-M), CAMUS dataset")
    print("COMET boundary metrics use image-based edge detection (Sobel)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="COMET Complete Evaluation (MOTFM-aligned)")
    parser.add_argument("--config_path", type=str, default="configs/comet_camus.yaml")
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                       help="One or more checkpoint paths to evaluate")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.model_paths[0]),
            "evaluation_results"
        )
    
    all_results = []
    for mp in args.model_paths:
        print(f"\n{'#'*70}")
        print(f"# Checkpoint: {os.path.basename(mp)}")
        print(f"{'#'*70}")
        
        res = evaluate_checkpoint(
            config_path=args.config_path,
            model_path=mp,
            num_samples=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            output_dir=args.output_dir,
            device=device,
        )
        all_results.append(res)
    
    # Final comparison table
    print_comparison_table(all_results)
    
    # Save combined results
    combined_path = os.path.join(args.output_dir, 'combined_results.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to: {combined_path}")
    
    print("\nAll evaluations complete!")


if __name__ == "__main__":
    main()
