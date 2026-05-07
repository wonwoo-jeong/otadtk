"""High-level ``OTAD`` class with FAD/KAD-compatible CLI and Python surface.

Usage example
-------------

>>> from otadtk import OTAD
>>> otad = OTAD(model="panns")           # auto-loads the agnostic adapter
>>> score = otad.score("ref/dir", "eval/dir")
>>> diag = otad.diagnose("ref/dir", "eval/dir")  # per-sample c_j + AUROC
>>> indiv_csv = otad.score_individual("ref/dir", "eval/dir", csv="per_file.csv")

The class is purposely modelled on ``kadtk.kad.KernelAudioDistance`` so users
who already work with ``kadtk`` can switch metrics with a one-line change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

from .adapter import RiemannianAdapter, load_adapter
from .emb_loader import EmbeddingLoader
from .model_loader import BaseEncoder, get_encoder
from .sinkhorn import sinkhorn_divergence

log = logging.getLogger(__name__)

PathLike = str | Path
Variant = Literal["agnostic", "native", "raw"]


# ---------------------------------------------------------------------------
# Functional API  (works on numpy embedding arrays).
# ---------------------------------------------------------------------------
def compute_otad(
    embeddings_ref: np.ndarray,
    embeddings_test: np.ndarray,
    *,
    adapter: RiemannianAdapter | None = None,
    epsilon: float = 0.1,
    return_sample_costs: bool = False,
    device: str | None = None,
):
    """Compute the OTAD score (debiased Sinkhorn divergence) between two
    embedding banks.  When ``return_sample_costs=True`` we also return the
    per-sample marginal transport cost ``c_j`` over the test set."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = torch.as_tensor(embeddings_ref, dtype=torch.float32, device=device)
    tst = torch.as_tensor(embeddings_test, dtype=torch.float32, device=device)

    if adapter is not None:
        adapter.eval().to(device)
        with torch.no_grad():
            ref = adapter(ref)
            tst = adapter(tst)

    score, T_xy, C_xy = sinkhorn_divergence(
        ref, tst, epsilon=epsilon, return_transport=True
    )
    score_value = float(score.item())
    if not return_sample_costs:
        return score_value

    cj = (T_xy * C_xy).sum(dim=0).detach().cpu().numpy()
    return score_value, cj


# ---------------------------------------------------------------------------
# Object-oriented API.
# ---------------------------------------------------------------------------
@dataclass
class OTADIndividualResult:
    file: Path
    cj: float


class OTAD:
    """Convenience class around ``compute_otad`` with embedding cache.

    Parameters
    ----------
    model : str
        Encoder name registered with otadtk (e.g. ``'panns'``, ``'panns_wglm'``,
        ``'vggish'``, ``'clap'``, ``'openl3'``, ``'audiomae'``, ``'ast'``,
        ``'beats'``, ``'encodec'``).
    variant : {"agnostic", "native", "raw"}, default ``"agnostic"``
        Adapter variant.  ``"agnostic"`` uses the contrastively-trained adapter
        (recommended for cross-condition decompositions and MOS evaluation),
        ``"native"`` uses the Sinkhorn-trained adapter, ``"raw"`` runs OTAD on
        the encoder's raw embeddings (no adapter).
    epsilon : float, default ``0.1``
        Sinkhorn regularisation.  See Section 5.6 of the paper for sweep
        guidance; the recommended range is ``[0.05, 0.10]``.
    device : str | None
        ``"cuda"`` (default if available) or ``"cpu"``.
    encoder : BaseEncoder | None
        Pre-instantiated encoder (overrides ``model``).
    adapter : RiemannianAdapter | None
        Pre-loaded adapter (overrides ``variant``-driven download).
    adapter_checkpoint : PathLike | None
        Local path to an adapter checkpoint.  Useful when the user retrains
        an adapter or ships a stripped distribution where the bundled
        checkpoints have been removed.
    force_recompute : bool, default ``False``
        Recompute the embedding cache even if a cached file exists.
    """

    def __init__(
        self,
        model: str = "panns",
        *,
        variant: Variant = "agnostic",
        epsilon: float = 0.1,
        device: str | None = None,
        encoder: BaseEncoder | None = None,
        adapter: RiemannianAdapter | None = None,
        adapter_checkpoint: PathLike | None = None,
        force_recompute: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = float(epsilon)
        self.variant: Variant = variant

        self.encoder: BaseEncoder = encoder or get_encoder(model, device=self.device)
        self.model_name: str = self.encoder.name

        if adapter is None and variant != "raw":
            adapter = load_adapter(
                self.encoder.name,
                variant,
                checkpoint_path=adapter_checkpoint,
                device=self.device,
            )
        self.adapter: RiemannianAdapter | None = adapter
        self.loader = EmbeddingLoader(self.encoder, force_recompute=force_recompute)

    # ------------------------------------------------------------------
    # Cache control.
    # ------------------------------------------------------------------
    def cache_embeddings(self, audio_dir: PathLike, *, workers: int = 1) -> list[Path]:
        return self.loader.cache_directory(Path(audio_dir), workers=workers)

    # ------------------------------------------------------------------
    # Distance.
    # ------------------------------------------------------------------
    def score(self, ref: PathLike, eval: PathLike, *, workers: int = 1) -> float:
        embs_ref, _ = self.loader.load_embeddings(Path(ref), workers=workers)
        embs_eval, _ = self.loader.load_embeddings(Path(eval), workers=workers)
        return compute_otad(
            embs_ref,
            embs_eval,
            adapter=self.adapter,
            epsilon=self.epsilon,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Per-sample diagnostics.
    # ------------------------------------------------------------------
    def diagnose(
        self,
        ref: PathLike,
        eval: PathLike,
        *,
        workers: int = 1,
    ):
        """Return a populated ``OTADDiagnostics`` object."""
        from .diagnostics import OTADDiagnostics

        embs_ref, _ = self.loader.load_embeddings(Path(ref), workers=workers)
        embs_eval, eval_files = self.loader.load_embeddings(Path(eval), workers=workers)
        score, cj = compute_otad(
            embs_ref,
            embs_eval,
            adapter=self.adapter,
            epsilon=self.epsilon,
            return_sample_costs=True,
            device=self.device,
        )
        return OTADDiagnostics(
            score=score,
            cj=cj,
            files=eval_files,
            model_name=self.model_name,
            variant=self.variant,
            epsilon=self.epsilon,
        )

    def score_individual(
        self,
        ref: PathLike,
        eval: PathLike,
        *,
        csv: PathLike | None = None,
        workers: int = 1,
    ) -> list[OTADIndividualResult]:
        """Return one ``OTADIndividualResult`` per evaluation file (sorted by
        ``cj`` ascending) and optionally write them to ``csv``."""
        diag = self.diagnose(ref, eval, workers=workers)
        results = sorted(
            (OTADIndividualResult(f, float(c)) for f, c in zip(diag.files, diag.cj)),
            key=lambda r: r.cj,
        )
        if csv is not None:
            csv_path = Path(csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w") as fp:
                fp.write("file,cj\n")
                for r in results:
                    fp.write(f"{r.file},{r.cj:.6f}\n")
            log.info(f"Wrote {len(results)} per-file scores to {csv_path}")
        return results
