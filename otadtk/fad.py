"""Fréchet Audio Distance reference implementation, kept aligned with the
classic Kilgour et al. definition: ``||mu_r - mu_t||^2 + Tr(Sigma_r + Sigma_t -
2*(Sigma_r^{1/2} Sigma_t Sigma_r^{1/2})^{1/2})``.

Used by OTAD as a baseline and inside the factorial decomposition (``cost_only``
condition).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg


def compute_fad(embeddings_ref: np.ndarray, embeddings_test: np.ndarray) -> float:
    mu_r, mu_t = embeddings_ref.mean(axis=0), embeddings_test.mean(axis=0)
    sigma_r = np.cov(embeddings_ref, rowvar=False)
    sigma_t = np.cov(embeddings_test, rowvar=False)

    diff = mu_r - mu_t
    covmean, _ = scipy.linalg.sqrtm(sigma_r @ sigma_t, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = float(diff @ diff + np.trace(sigma_r + sigma_t - 2.0 * covmean))
    return max(fad, 0.0)
