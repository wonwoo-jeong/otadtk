"""OTAD's per-sample diagnostics.

The ``OTADDiagnostics`` object exposes everything otadtk knows at the
per-evaluation-sample level: the marginal transport cost ``c_j``, the
file paths the costs correspond to, plus convenience summaries
(``top_k``, ``auroc``, ``separation_ratio``, ``to_csv``).

We deliberately avoid bundling any plotting code: visualisation is
better delegated to whichever stack the caller already uses
(matplotlib, seaborn, plotly, …).  All raw numbers are available as
plain numpy arrays so ``ax.hist(diag.cj)`` is one line away.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - sklearn is in our dependency set
    roc_auc_score = None  # type: ignore[assignment]


@dataclass
class OTADDiagnostics:
    """Result of ``OTAD.diagnose``.

    Attributes
    ----------
    score : float
        Aggregate OTAD score (debiased Sinkhorn divergence).
    cj : np.ndarray
        Per-evaluation-sample marginal transport cost; shape ``(M,)``.
    files : list[Path]
        Evaluation-file paths corresponding to ``cj``.
    model_name, variant, epsilon : str / float
        Provenance metadata.
    """

    score: float
    cj: np.ndarray
    files: list[Path]
    model_name: str
    variant: str
    epsilon: float

    # ------------------------------------------------------------------
    # Summary helpers.
    # ------------------------------------------------------------------
    def top_k(self, k: int = 10, *, ascending: bool = False) -> list[tuple[Path, float]]:
        """Return the ``k`` evaluation files with the largest (default) or
        smallest ``cj``.  The "largest" tail is what reviewers typically want
        to listen to: those samples drag OTAD up the most."""
        order = np.argsort(self.cj)
        if not ascending:
            order = order[::-1]
        return [(self.files[i], float(self.cj[i])) for i in order[:k]]

    # ------------------------------------------------------------------
    # Label-based metrics  (optional).
    # ------------------------------------------------------------------
    def separation_ratio(self, contaminated_mask: Sequence[bool]) -> float:
        """``mean(cj | contaminated) / mean(cj | clean)``.

        ``contaminated_mask`` is a boolean iterable of length ``M`` indicating
        which evaluation files are known contaminations.
        """
        mask = np.asarray(contaminated_mask, dtype=bool)
        if mask.size != self.cj.size:
            raise ValueError("mask size mismatch")
        if mask.sum() == 0 or (~mask).sum() == 0:
            raise ValueError("Both contaminated and clean subsets must be non-empty")
        return float(self.cj[mask].mean() / self.cj[~mask].mean())

    def auroc(self, contaminated_mask: Sequence[bool]) -> float:
        """AUROC of ``cj`` as a contamination detector."""
        if roc_auc_score is None:
            raise RuntimeError("sklearn is required for diagnostics.auroc")
        mask = np.asarray(contaminated_mask, dtype=int)
        return float(roc_auc_score(mask, self.cj))

    # ------------------------------------------------------------------
    # IO.
    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fp:
            fp.write("file,cj\n")
            for f, c in zip(self.files, self.cj):
                fp.write(f"{f},{c:.6f}\n")
        return path
