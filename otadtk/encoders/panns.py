"""PANNs (CNN14) encoder — d=2048, TTA community standard FAD encoder."""

import numpy as np
import torch
from ..model_loader import BaseEncoder
from ..model_loader import register_encoder


@register_encoder("panns")
class PANNsEncoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "panns"
        self.target_sr = 32000
        self.embedding_dim = 2048
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from panns_inference import AudioTagging
            self._tagger = AudioTagging(
                checkpoint_path=None, device=self.device
            )
            self._model = self._tagger
        except ImportError:
            raise ImportError(
                "Install panns_inference: pip install panns-inference"
            )

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        self._load()
        wav = self.pad_if_needed(wav, sr)
        import torchaudio
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        audio_np = wav.squeeze().cpu().numpy()[None, :]
        _, emb = self._tagger.inference(audio_np)
        return emb.squeeze()
