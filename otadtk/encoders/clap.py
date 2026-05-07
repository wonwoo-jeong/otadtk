"""CLAP encoder — d=512, cross-modal contrastive."""

import numpy as np
import torch
from ..model_loader import BaseEncoder
from ..model_loader import register_encoder


@register_encoder("clap")
class CLAPEncoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "clap"
        self.target_sr = 48000
        self.embedding_dim = 512
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import laion_clap
        self._model = laion_clap.CLAP_Module(enable_fusion=False)
        self._model.load_ckpt()
        self._model.eval()
        self._model.to(self.device)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        self._load()
        wav = self.pad_if_needed(wav, sr)
        import torchaudio
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        audio_np = wav.squeeze().cpu().numpy()
        emb = self._model.get_audio_embedding_from_data(
            [audio_np], use_tensor=False
        )
        return emb.squeeze()
