"""
Contrastive Warm-Start Flow Matching (COMET) Loss Components
=============================================================

This module combines techniques from two published works, adapted for
mask-conditional medical image flow matching:

1. **Contrastive Flow Loss** — adapted from DeltaFM (ICCV 2025):
   Stoica et al., "Contrastive Flow Matching"
   Original code: https://github.com/gstoica27/DeltaFM
   File: triplet_loss.py -> TripletSILoss.compute_triplet_loss_efficiently
   Key idea: loss = pos_error - temperature * neg_error, where negatives
   are created by shuffling target velocities within the batch.

2. **Warm-Start Coupling** — inspired by Conditional Prior FM (TMLR):
   Salama et al., "Designing a Conditional Prior Distribution for
   Flow-Based Generative Models"
   Original code: https://github.com/MoSalama98/conditional-prior-flow-matching
   File: train.py -> x0 = decoder(condition, z) + randn * sigma
   Key idea: replace isotropic Gaussian x_0 with a condition-specific
   structured prior to shorten the transport path.

   Our adaptation: instead of a learned decoder, we use Gaussian-blurred
   segmentation masks as the spatial prior (no extra network needed).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Utilities copied/adapted from DeltaFM (triplet_loss.py, loss.py)
###############################################################################

def mean_flat(x):
    """Take the mean over all non-batch dimensions.

    Copied from: DeltaFM/loss.py::mean_flat
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def _random_negative_indices(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample a random permutation that avoids self-matching.

    Adapted from: DeltaFM/triplet_loss.py::compute_triplet_loss_efficiently
    (lines creating `choices` tensor).
    """
    choices = torch.tile(torch.arange(batch_size), (batch_size, 1)).to(device)
    choices.fill_diagonal_(-1.0)
    choices = choices.sort(dim=1)[0][:, 1:]  # remove diagonal -1
    choices = choices[
        torch.arange(batch_size, device=device),
        torch.randint(0, batch_size - 1, (batch_size,), device=device),
    ]
    return choices


###############################################################################
# Warm-Start Utilities (inspired by ConditionalPriorFM/train.py)
###############################################################################

def gaussian_blur_2d(
    x: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Apply Gaussian blur to a batch of 2-D images.

    Standard utility — used to convert segmentation masks into smooth
    spatial priors for warm-start coupling.

    Args:
        x: Input tensor ``[B, C, H, W]``.
        kernel_size: Gaussian kernel size (odd).
        sigma: Gaussian standard deviation.

    Returns:
        Blurred tensor of the same shape.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    channels = x.shape[1]

    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-coords.pow(2) / (2.0 * sigma ** 2))
    g = g / g.sum()

    kernel_2d = g.outer(g)  # [K, K]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(channels, -1, -1, -1)

    padding = kernel_size // 2
    return F.conv2d(x, kernel_2d, padding=padding, groups=channels)


def create_warm_start_noise(
    masks: torch.Tensor,
    noise: torch.Tensor,
    alpha: float = 0.3,
    blur_kernel_size: int = 15,
    blur_sigma: float = 5.0,
) -> torch.Tensor:
    """Create warm-started initial noise from segmentation masks.

    Inspired by ConditionalPriorFM/train.py (lines 180-182)::

        x0 = decoder(clip_texts, zi)                        # condition-specific prior
        x0 = x0 + (torch.randn_like(x0) * sigma)           # add noise

    Our adaptation for mask-conditional medical imaging:
    instead of a learned decoder, we use Gaussian-blurred masks
    as the spatial prior::

        x0 = alpha * blur(mask) + sqrt(1 - alpha^2) * noise

    This preserves approximate unit variance while embedding structural
    information from the segmentation mask.

    Args:
        masks: Segmentation masks ``[B, C_mask, H, W]`` in [0, 1].
        noise: Standard Gaussian noise ``[B, C_img, H, W]``.
        alpha: Warm-start strength in [0, 1].
        blur_kernel_size: Gaussian blur kernel size.
        blur_sigma: Gaussian blur sigma.

    Returns:
        Warm-started noise ``[B, C_img, H, W]``.
    """
    blurred = gaussian_blur_2d(masks, kernel_size=blur_kernel_size, sigma=blur_sigma)

    # Match channels if mask and image have different channel counts
    if blurred.shape[1] != noise.shape[1]:
        blurred = blurred.mean(dim=1, keepdim=True).expand_as(noise)

    # Variance-preserving coupling (same spirit as ConditionalPriorFM's sigma parameter)
    scale_noise = math.sqrt(1.0 - alpha ** 2)
    x_0 = alpha * blurred + scale_noise * noise
    return x_0


###############################################################################
# Contrastive Flow Loss — Cosine-InfoNCE in Velocity Space
###############################################################################
#
# Evolution from DeltaFM's formulation:
# - DeltaFM uses: loss = pos_error - temp * neg_error  (unbounded below)
# - L2-InfoNCE causes velocity collapse: model shrinks velocity magnitude
#   to satisfy contrastive objective (discriminability), leading to
#   systematically under-transported outputs (dark/biased images).
#
# Our formulation: Cosine-similarity InfoNCE (same family as CLIP/SimCLR).
# - L2-normalizes velocity vectors before computing similarity.
# - logit[i,j] = cos(v_pred[i], target[j]) / tau
# - Loss = CrossEntropy(logits, labels=arange(B))
#
# Why cosine similarity solves the collapse:
# - Scale-invariant: only the DIRECTION of v_pred matters, not magnitude.
# - Model cannot shrink velocities to game the contrastive loss.
# - MSE loss controls the magnitude; InfoNCE controls the direction.
# - Clean separation of concerns → no interference between objectives.
#
# Properties:
# - Always non-negative, bounded in [0, log(B)]  ✓
# - Scale-invariant (prevents velocity collapse)  ✓
# - Uses ALL negatives in batch (not just 1)      ✓
# - Well-studied (CLIP, SimCLR, MoCo foundations)  ✓
#
# Novel contribution: Cosine-InfoNCE has never been applied to the velocity
# prediction space of conditional flow matching for medical imaging.
###############################################################################

class ContrastiveFlowLoss(nn.Module):
    """Cosine-similarity InfoNCE contrastive loss in velocity space.

    Inspired by the contrastive principle from DeltaFM (Stoica et al., ICCV 2025)
    but reformulated using cosine similarity (as in CLIP/SimCLR) to prevent
    velocity magnitude collapse.

    For a batch of B samples, the predicted velocity direction should be most
    aligned (highest cosine similarity) with its ground-truth target compared
    to all other targets::

        logits[i, j] = cos_sim(v_pred[i], target[j]) / tau
        loss = CrossEntropy(logits, labels=arange(B))

    By operating on L2-normalized vectors, this loss:
    - Only constrains velocity DIRECTION (condition-specificity)
    - Does NOT constrain velocity MAGNITUDE (left to MSE loss)
    - Prevents the model from shrinking velocities to satisfy contrastive obj

    Args:
        temperature: Softmax temperature controlling discrimination sharpness.
            For cosine similarity in [-1, 1], typical range is [0.05, 0.5].
            Default 0.07 (following CLIP).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # Learnable temperature can be beneficial but we keep it fixed for simplicity
        self.temperature = temperature

    def forward(
        self,
        v_pred: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute cosine-InfoNCE contrastive flow loss.

        Args:
            v_pred: Model-predicted velocity ``[B, C, H, W]``.
            target_velocity: Ground-truth velocity ``[B, C, H, W]``.

        Returns:
            (loss, diagnostics) where loss is a scalar tensor >= 0.
        """
        B = v_pred.shape[0]
        zero_diag = {
            "contrastive/loss": torch.tensor(0.0),
            "contrastive/cos_pos": torch.tensor(1.0),
            "contrastive/cos_neg": torch.tensor(0.0),
            "contrastive/accuracy": torch.tensor(1.0),
        }
        if B <= 1:
            return torch.tensor(0.0, device=v_pred.device, requires_grad=True), zero_diag

        # ---- Cast to FP32 for numerical stability ----
        v_flat = v_pred.float().flatten(1)              # [B, D]
        target_flat = target_velocity.float().flatten(1) # [B, D]

        # ---- L2 normalize (direction only, magnitude discarded) ----
        v_norm = F.normalize(v_flat, p=2, dim=1)       # [B, D], unit vectors
        t_norm = F.normalize(target_flat, p=2, dim=1)   # [B, D], unit vectors

        # ---- Cosine similarity matrix ----
        # sim[i,j] = cos(v_pred[i], target[j])  ∈ [-1, 1]
        sim_matrix = v_norm @ t_norm.T  # [B, B]

        # ---- InfoNCE logits ----
        logits = sim_matrix / self.temperature  # [B, B]
        labels = torch.arange(B, device=v_pred.device)

        # ---- Cross-entropy loss ----
        contrastive_loss = F.cross_entropy(logits, labels)

        # ---- Diagnostics ----
        with torch.no_grad():
            cos_pos = sim_matrix.diag().mean()  # mean cosine with correct target
            mask = ~torch.eye(B, dtype=torch.bool, device=v_pred.device)
            cos_neg = sim_matrix[mask].mean()   # mean cosine with wrong targets
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean()

        diagnostics = {
            "contrastive/loss": contrastive_loss.detach(),
            "contrastive/cos_pos": cos_pos,
            "contrastive/cos_neg": cos_neg,
            "contrastive/accuracy": accuracy,
        }
        return contrastive_loss, diagnostics
