"""Encoder ABC and registry, modelled on the kadtk ``ModelLoader`` interface
but with a slimmer ``encode(wav, sr)`` API that mirrors otadtk's training code.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Callable, Dict

import numpy as np
import torch


class BaseEncoder(ABC):
    """All encoders subclass this and implement ``encode``.

    Attributes
    ----------
    name : str
        Registry key (also used in cache paths).
    target_sr : int
        Sample rate the encoder was trained at; otadtk resamples on demand.
    embedding_dim : int
        Dimensionality of the global embedding returned by ``encode``.
    """

    MIN_AUDIO_SECONDS = 1.5

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.name: str = "base"
        self.target_sr: int = 16000
        self.embedding_dim: int = 0

    def pad_if_needed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        min_samples = int(self.MIN_AUDIO_SECONDS * sr)
        if wav.shape[-1] < min_samples:
            pad_len = min_samples - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        return wav

    @abstractmethod
    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        """Encode a single waveform.

        Parameters
        ----------
        wav : torch.Tensor of shape (1, T)
        sr : int

        Returns
        -------
        numpy array of shape (embedding_dim,)
        """

    def encode_batch(self, wavs: list[torch.Tensor], sr: int) -> np.ndarray:
        """Encode a list of waveforms.  Default: sequential fallback."""
        return np.stack([self.encode(w, sr) for w in wavs])

    def to(self, device: str) -> "BaseEncoder":
        self.device = device
        return self


# ----------------------------------------------------------------------
# Registry & lazy import.
# ----------------------------------------------------------------------
ENCODER_REGISTRY: Dict[str, Callable[..., BaseEncoder]] = {}


def register_encoder(name: str):
    """Decorator that registers an encoder class under ``name``."""

    def wrapper(cls):
        ENCODER_REGISTRY[name] = cls
        return cls

    return wrapper


_LAZY_MODULE_MAP: Dict[str, str] = {
    "vggish": ".encoders.vggish",
    "panns": ".encoders.panns",
    "panns_wglm": ".encoders.panns_wglm",
    "clap": ".encoders.clap",
    "openl3": ".encoders.openl3",
    "audiomae": ".encoders.audiomae",
    "ast": ".encoders.ast",
    "beats": ".encoders.beats",
    "encodec": ".encoders.encodec",
}


def _ensure_imported(name: str) -> None:
    if name in ENCODER_REGISTRY:
        return
    if name in _LAZY_MODULE_MAP:
        importlib.import_module(_LAZY_MODULE_MAP[name], package="otadtk")


def get_encoder(name: str, device: str = "cuda", **kwargs) -> BaseEncoder:
    """Instantiate the encoder registered under ``name``."""
    _ensure_imported(name)
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {sorted(_LAZY_MODULE_MAP)}"
        )
    return ENCODER_REGISTRY[name](device=device, **kwargs)


def list_models() -> list[str]:
    """Names of all encoders that otadtk can load (lazily)."""
    return sorted(_LAZY_MODULE_MAP.keys())
