"""EnCodec encoder — d=128, neural audio compression."""

import numpy as np
import torch
from ..model_loader import BaseEncoder
from ..model_loader import register_encoder


@register_encoder("encodec")
class EnCodecEncoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "encodec"
        self.target_sr = 24000
        self.embedding_dim = 128
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from encodec import EncodecModel
        self._model = EncodecModel.encodec_model_24khz()
        self._model.set_target_bandwidth(6.0)
        self._model.eval()
        self._model.to(self.device)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        self._load()
        wav = self.pad_if_needed(wav, sr)
        import torchaudio
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)
        emb = self._model.encoder(wav)
        emb = emb.mean(dim=-1).squeeze()
        return emb.cpu().numpy()
