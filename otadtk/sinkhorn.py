"""Log-domain Sinkhorn primitives used by OTAD.

The implementation is numerically stable for the large pairwise-cost matrices
produced by audio embeddings (`d` up to a few thousand, `n` up to a few thousand
samples).  The main exposed symbols are:

    - ``log_sinkhorn_transport(C, epsilon)``  -> transport plan T (n, m)
    - ``log_sinkhorn_cost(C, epsilon)``       -> scalar OT_eps cost
    - ``sinkhorn_divergence(x, y, epsilon)``  -> debiased S_eps(x, y)
"""
from __future__ import annotations

import numpy as np
import torch


def log_sinkhorn_transport(
    C: torch.Tensor,
    epsilon: float,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Return the entropic OT transport plan T (n, m) under uniform marginals."""
    n, m = C.shape
    log_mu = torch.full((n,), -float(np.log(n)), device=C.device, dtype=C.dtype)
    log_nu = torch.full((m,), -float(np.log(m)), device=C.device, dtype=C.dtype)
    f = torch.zeros(n, device=C.device, dtype=C.dtype)
    g = torch.zeros(m, device=C.device, dtype=C.dtype)

    for _ in range(max_iter):
        f_prev = f.clone()
        f = -epsilon * torch.logsumexp((g[None, :] - C) / epsilon + log_nu[None, :], dim=1)
        g = -epsilon * torch.logsumexp((f[:, None] - C) / epsilon + log_mu[:, None], dim=0)
        if (f - f_prev).abs().max() < tol:
            break

    log_T = (f[:, None] + g[None, :] - C) / epsilon + log_mu[:, None] + log_nu[None, :]
    return torch.exp(log_T)


def log_sinkhorn_cost(
    C: torch.Tensor,
    epsilon: float,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> torch.Tensor:
    T = log_sinkhorn_transport(C, epsilon, max_iter=max_iter, tol=tol)
    return (T * C).sum()


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-6,
    return_transport: bool = False,
):
    """Debiased Sinkhorn divergence ``S_eps(x, y) = OT(x,y) - 0.5 OT(x,x) - 0.5 OT(y,y)``.

    Cost is squared Euclidean.  When ``return_transport=True`` we also return
    the cross-term transport plan, which is the object that supports
    OTAD's per-sample diagnostics.
    """
    C_xy = torch.cdist(x, y, p=2).pow(2)
    T_xy = log_sinkhorn_transport(C_xy, epsilon, max_iter=max_iter, tol=tol)
    ot_xy = (T_xy * C_xy).sum()

    ot_xx = log_sinkhorn_cost(torch.cdist(x, x, p=2).pow(2), epsilon, max_iter, tol)
    ot_yy = log_sinkhorn_cost(torch.cdist(y, y, p=2).pow(2), epsilon, max_iter, tol)

    score = ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
    if return_transport:
        return score, T_xy, C_xy
    return score
