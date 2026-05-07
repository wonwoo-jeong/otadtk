"""Riemannian ground-metric adapter (`g_theta(z) = z + f_theta(z)`).

The adapter is a residual bottleneck MLP whose Jacobian J = I + J_{f_theta}
defines a learned local Mahalanobis metric on the embedding space. Pretrained
checkpoints for the encoders bundled with otadtk ship inside the wheel under
``otadtk/checkpoints/`` (see ``_checkpoints.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from ._checkpoints import resolve_adapter_checkpoint

Variant = Literal["agnostic", "native", "raw"]


class RiemannianAdapter(nn.Module):
    """Residual bottleneck MLP: ``g_theta(z) = z + f_theta(z)``.

    Parameters
    ----------
    input_dim : int
        Encoder embedding dimension (e.g. 128, 512, 768, 2048).
    bottleneck_ratio : int, default 4
        Hidden width is ``input_dim // bottleneck_ratio``.
    dropout : float, default 0.1
        Dropout in the bottleneck.
    """

    def __init__(self, input_dim: int, bottleneck_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(input_dim // bottleneck_ratio, 1)
        self.f_theta = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        nn.init.zeros_(self.f_theta[-1].weight)
        nn.init.zeros_(self.f_theta[-1].bias)
        self.input_dim = input_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.f_theta(z)


def load_adapter(
    encoder_name: str,
    variant: Variant = "agnostic",
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    bottleneck_ratio: int = 4,
    dropout: float = 0.1,
) -> RiemannianAdapter | None:
    """Load a pretrained adapter for ``encoder_name`` and ``variant``.

    Returns ``None`` for ``variant='raw'`` (OTAD on raw embeddings).

    The checkpoint is resolved in this order:
      1. Explicit ``checkpoint_path`` argument, if given.
      2. ``OTADTK_CHECKPOINTS`` environment variable (a local cache directory).
      3. Package-bundled files under ``otadtk/checkpoints/`` (default install).

    Anything that fails falls back to raising a descriptive ``RuntimeError`` so
    the user can decide whether to train an adapter from scratch with the
    factorial scripts shipped in the source repository.
    """
    if variant == "raw":
        return None

    ckpt = resolve_adapter_checkpoint(encoder_name, variant, override=checkpoint_path)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    adapter_state = state["adapter_state_dict"] if "adapter_state_dict" in state else state

    first_weight = next(
        (v for k, v in adapter_state.items() if k.endswith(".weight") and v.dim() == 2),
        None,
    )
    if first_weight is None:
        raise RuntimeError(f"Could not infer input_dim from {ckpt}")
    input_dim = first_weight.shape[1]

    adapter = RiemannianAdapter(
        input_dim=input_dim, bottleneck_ratio=bottleneck_ratio, dropout=dropout
    ).to(device)
    adapter.load_state_dict(adapter_state)
    adapter.eval()
    return adapter
