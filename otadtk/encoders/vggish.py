"""VGGish encoder — d=128, original FAD encoder."""

import numpy as np
import torch
from ..model_loader import BaseEncoder
from ..model_loader import register_encoder


@register_encoder("vggish")
class VGGishEncoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "vggish"
        self.target_sr = 16000
        self.embedding_dim = 128
        self._model = None
        self._preprocess = None

    def _load(self):
        if self._model is not None:
            return
        self._model = torch.hub.load("harritaylor/torchvggish", "vggish")
        self._model.preprocess = False
        self._model.postprocess = False
        self._model.eval()
        self._model.to(self.device)

        from pathlib import Path
        import sys
        hub_dir = Path(torch.hub.get_dir()) / "harritaylor_torchvggish_master"
        sys.path.insert(0, str(hub_dir))
        from torchvggish.vggish_input import waveform_to_examples
        self._preprocess = waveform_to_examples

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        self._load()
        audio_np = wav.squeeze().cpu().numpy()
        min_len = int(sr * 1.1)  # VGGish needs >= ~1s
        if audio_np.shape[-1] < min_len:
            audio_np = np.pad(audio_np, (0, min_len - audio_np.shape[-1]))
        mel_input = self._preprocess(audio_np, sr)
        mel_input = mel_input.to(self.device)
        emb = self._model(mel_input)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)
        return emb.cpu().numpy()
