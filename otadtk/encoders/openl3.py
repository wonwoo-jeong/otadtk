"""OpenL3 encoder — d=512, Audio-Visual Embedding (torchopenl3 backend)."""

import numpy as np
import torch
from ..model_loader import BaseEncoder
from ..model_loader import register_encoder


@register_encoder("openl3")
class OpenL3Encoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "openl3"
        self.target_sr = 48000
        self.embedding_dim = 512
        self._model = None
        self._lib = None

    def _load(self):
        if self._model is not None:
            return
        import torchopenl3
        self._lib = torchopenl3
        self._model = torchopenl3.core.load_audio_embedding_model(
            "mel256", "music", 512
        )
        self._model.eval()
        self._model.to(self.device)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        self._load()
        audio_np = wav.squeeze().cpu().numpy()
        min_len = int(sr * 1.5)
        if audio_np.shape[-1] < min_len:
            audio_np = np.pad(audio_np, (0, min_len - audio_np.shape[-1]))
        emb, _ = self._lib.get_audio_embedding(
            audio_np, sr, model=self._model,
            content_type="music", embedding_size=512,
        )
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        while emb.ndim > 1:
            emb = emb.mean(axis=0) if emb.shape[0] > 1 else emb.squeeze(0)
        return emb.astype(np.float32)
