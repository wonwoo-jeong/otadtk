"""Embedding loader with on-disk cache.

Each ``(audio_dir, model_name)`` pair gets a cache layout that matches kadtk:

    <audio_dir>/embedding/<model_name>/<basename>.npy

so otadtk and kadtk can interoperate on the same dataset directory.

We deliberately avoid the SoX/ffmpeg conversion step that kadtk runs as a
preprocessing pass — otadtk resamples on the fly inside each encoder's
``encode``.  This makes otadtk usable inside Docker images that have no system
audio toolchain.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .model_loader import BaseEncoder

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}


def _audio_files(directory: Path) -> list[Path]:
    if directory.is_file():
        return [directory]
    files = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in AUDIO_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No supported audio files under {directory}")
    return files


def _cache_path(audio_path: Path, model_name: str) -> Path:
    audio_path = Path(audio_path)
    parent = audio_path.parent if audio_path.is_file() else audio_path
    return parent / "embedding" / model_name / f"{audio_path.stem}.npy"


def _load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load audio to mono ``(1, T)`` tensor at ``target_sr`` using soundfile + librosa."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return torch.from_numpy(data).unsqueeze(0)


class EmbeddingLoader:
    """Caches encoder embeddings to disk; reuses across runs."""

    def __init__(self, encoder: BaseEncoder, *, force_recompute: bool = False):
        self.encoder = encoder
        self.force_recompute = force_recompute

    # ------------------------------------------------------------------
    # Single-file API.
    # ------------------------------------------------------------------
    def cache_file(self, audio_path: Path) -> Path:
        cache = _cache_path(audio_path, self.encoder.name)
        if cache.exists() and not self.force_recompute:
            return cache
        cache.parent.mkdir(parents=True, exist_ok=True)
        wav = _load_audio(audio_path, self.encoder.target_sr)
        emb = self.encoder.encode(wav, self.encoder.target_sr)
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        np.save(cache, emb.astype(np.float32))
        return cache

    def read(self, audio_path: Path) -> np.ndarray:
        cache = _cache_path(audio_path, self.encoder.name)
        if not cache.exists():
            cache = self.cache_file(audio_path)
        return np.load(cache).astype(np.float32)

    # ------------------------------------------------------------------
    # Directory-level API used by ``OTAD``.
    # ------------------------------------------------------------------
    def cache_directory(
        self, directory: Path, *, workers: int = 1, desc: str | None = None
    ) -> list[Path]:
        files = _audio_files(directory)
        desc = desc or f"Encoding [{self.encoder.name}] {directory}"
        if workers <= 1:
            for f in tqdm(files, desc=desc):
                self.cache_file(f)
            return files
        # NB: most encoders are not thread-safe (CUDA + python state); keep workers
        # for I/O-bound cache hits only.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self.cache_file, f) for f in files]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=desc):
                pass
        return files

    def load_embeddings(
        self, directory: Path, *, workers: int = 1
    ) -> tuple[np.ndarray, list[Path]]:
        files = self.cache_directory(directory, workers=workers)
        embs = np.stack([self.read(f) for f in files])
        return embs.astype(np.float32), files

    def load_files(self, files: Sequence[Path]) -> np.ndarray:
        return np.stack([self.read(f) for f in files]).astype(np.float32)
