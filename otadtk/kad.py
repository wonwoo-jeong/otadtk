"""Kernel Audio Distance baseline, aligned with the kadtk reference.

We follow the kadtk convention for the bandwidth heuristic — the *median
pairwise Euclidean distance over the test set* — to keep our KAD numbers
directly comparable with values produced by ``kadtk.kad.calc_kernel_audio_distance``.
The output value matches kadtk's ``SCALE_FACTOR=100`` * MMD^2 when ``scale=True``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def median_pairwise_distance(x: torch.Tensor) -> float:
    return float(torch.median(torch.pdist(x.float())).item())


def compute_kad(
    embeddings_ref: np.ndarray,
    embeddings_test: np.ndarray,
    bandwidth: float | None = None,
    kernel: Literal["gaussian", "iq", "imq"] = "gaussian",
    scale: bool = True,
    eps: float = 1e-8,
    device: str | None = None,
) -> float:
    """Unbiased squared MMD with an RBF/IQ/IMQ kernel.

    By default the bandwidth is the median pairwise Euclidean distance of the
    *test (eval) set*, matching kadtk; pass ``bandwidth=...`` to override.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(embeddings_ref, dtype=torch.float32, device=device)
    y = torch.as_tensor(embeddings_test, dtype=torch.float32, device=device)

    if bandwidth is None:
        bandwidth = median_pairwise_distance(y)
        if bandwidth < 1e-8:
            bandwidth = 1.0

    gamma = 1.0 / (2.0 * bandwidth ** 2 + eps)
    if kernel == "gaussian":
        k = lambda d2: torch.exp(-gamma * d2)
    elif kernel == "iq":
        k = lambda d2: 1.0 / (1.0 + gamma * d2)
    elif kernel == "imq":
        k = lambda d2: 1.0 / torch.sqrt(1.0 + gamma * d2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    n_x, n_y = x.shape[0], y.shape[0]
    d2_xx = torch.cdist(x, x, p=2).pow(2)
    d2_yy = torch.cdist(y, y, p=2).pow(2)
    d2_xy = torch.cdist(x, y, p=2).pow(2)

    k_xx = k(d2_xx)
    k_yy = k(d2_yy)
    k_xx = k_xx - torch.diag(torch.diagonal(k_xx))
    k_yy = k_yy - torch.diag(torch.diagonal(k_yy))
    mmd2 = (
        k_xx.sum() / (n_x * (n_x - 1))
        + k_yy.sum() / (n_y * (n_y - 1))
        - 2.0 * k(d2_xy).mean()
    )
    out = float(mmd2.item())
    return 100.0 * out if scale else out
