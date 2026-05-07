"""PANNs Wavegram-LogMel-Cnn14 (WGLM) encoder — d=2048, sr=32k.

Identical backbone to the variant used by KAD (Chung et al., 2025) and
PANNs paper (Kong et al., 2020). Concatenates raw-waveform Wavegram
features with log-Mel features for stronger perceptual signal than CNN14.

Checkpoint: Wavegram_Logmel_Cnn14_mAP=0.439.pth (Zenodo 3987831).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ..model_loader import BaseEncoder
from ..model_loader import register_encoder


_DEFAULT_CKPT = Path.home() / "panns_data" / "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
_ZENODO_URL = (
    "https://zenodo.org/records/3987831/files/"
    "Wavegram_Logmel_Cnn14_mAP%3D0.439.pth"
)


@register_encoder("panns_wglm")
class PANNsWGLMEncoder(BaseEncoder):
    """Wavegram-LogMel-CNN14 PANNs variant (d=2048)."""

    def __init__(self, device: str = "cuda", checkpoint_path: str | None = None):
        super().__init__(device)
        self.name = "panns_wglm"
        self.target_sr = 32000
        self.embedding_dim = 2048
        self._ckpt = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CKPT
        self._model = None

    def _ensure_checkpoint(self) -> None:
        if self._ckpt.exists() and self._ckpt.stat().st_size > 3e8:
            return
        self._ckpt.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(_ZENODO_URL, self._ckpt)

    def _load(self) -> None:
        if self._model is not None:
            return
        self._ensure_checkpoint()
        from ._panns_wglm.models import Wavegram_Logmel_Cnn14
        net = Wavegram_Logmel_Cnn14(
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=527,
        )
        state = torch.load(self._ckpt, map_location=self.device, weights_only=False)
        net.load_state_dict(state["model"])
        net.eval()
        net.to(self.device)
        self._model = net

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        self._load()
        wav = self.pad_if_needed(wav, sr)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        x = wav.to(self.device).float()
        out = self._model(x)
        emb = out["embedding"].squeeze(0).detach().cpu().numpy()
        return emb.astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, wavs: list[torch.Tensor], sr: int,
                     max_seconds: float = 5.0,
                     micro_batch: int = 16) -> np.ndarray:
        """Vectorised encode that pads waveforms to a common length and runs
        forward in micro-batches to bound memory usage.

        Long clips are randomly cropped to ``max_seconds`` before stacking;
        this is critical during adapter training because Wavegram_Logmel_Cnn14
        materialises (B, 64, T)-shaped activations that grow linearly in T.
        """
        self._load()
        max_samples = int(max_seconds * self.target_sr)
        prepped = []
        for w in wavs:
            w = self.pad_if_needed(w, sr)
            if sr != self.target_sr:
                w = torchaudio.functional.resample(w, sr, self.target_sr)
            if w.dim() > 1:
                w = w.mean(dim=0)
            w = w.float().contiguous()
            if w.shape[-1] > max_samples:
                start = int(torch.randint(0, w.shape[-1] - max_samples + 1, (1,)).item())
                w = w[start : start + max_samples]
            prepped.append(w)
        max_len = max(int(w.shape[-1]) for w in prepped)
        outs = []
        for i in range(0, len(prepped), micro_batch):
            chunk = prepped[i : i + micro_batch]
            padded = torch.zeros(len(chunk), max_len, dtype=torch.float32)
            for j, w in enumerate(chunk):
                padded[j, : w.shape[-1]] = w
            out = self._model(padded.to(self.device))
            outs.append(out["embedding"].detach().cpu())
        return torch.cat(outs, dim=0).numpy().astype(np.float32)
