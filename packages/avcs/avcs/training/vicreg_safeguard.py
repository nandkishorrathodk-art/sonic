"""VICReg variance safeguard for AVCS training.

Use the *pre-L2-normalization* projection for variance regularization, and use
its L2-normalized form for contrastive similarity/retrieval.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def normalize_embedding(pre_normalized: Tensor, eps: float = 1e-8) -> Tensor:
    """Return unit-L2 embeddings for contrastive similarity or retrieval."""
    if pre_normalized.ndim != 2:
        raise ValueError("expected a [batch, embedding_dim] tensor")
    if pre_normalized.shape[0] < 1 or pre_normalized.shape[1] < 1:
        raise ValueError("batch and embedding dimensions must be non-empty")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not torch.isfinite(pre_normalized).all():
        raise ValueError("pre_normalized contains NaN or Inf")
    return F.normalize(pre_normalized, p=2, dim=-1, eps=eps)


def vicreg_variance_loss(
    pre_normalized: Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> Tensor:
    """Penalize low per-dimension batch spread before L2 normalization.

    Args:
        pre_normalized: Tensor shaped [batch, embedding_dim], before F.normalize.
        target_std: Minimum target standard deviation for each dimension.
        eps: Stability epsilon added inside the square root.

    Returns:
        Scalar mean ReLU penalty. This is the VICReg variance term only; it is
        not a complete VICReg objective (which also has invariance/covariance
        terms) and does not replace AVCS's other task losses.
    """
    if pre_normalized.ndim != 2:
        raise ValueError("expected a [batch, embedding_dim] tensor")
    batch_size, embedding_dim = pre_normalized.shape
    if batch_size < 2:
        raise ValueError("variance regularization requires batch_size >= 2")
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")
    if target_std <= 0:
        raise ValueError("target_std must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not torch.isfinite(pre_normalized).all():
        raise ValueError("pre_normalized contains NaN or Inf")

    # Use population variance for a predictable finite-batch definition.
    std = torch.sqrt(torch.var(pre_normalized, dim=0, unbiased=False) + eps)
    return F.relu(float(target_std) - std).mean()
