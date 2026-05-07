"""AudioMAE encoder — d=768, masked reconstruction pre-training."""

import numpy as np
import torch
from ..model_loader import BaseEncoder
from ..model_loader import register_encoder


@register_encoder("audiomae")
class AudioMAEEncoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "audiomae"
        self.target_sr = 16000
        self.embedding_dim = 768
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoFeatureExtractor
        model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        # AudioMAE uses AST-compatible architecture; swap with actual AudioMAE
        # checkpoint when available. For now, use AST as proxy.
        self._processor = AutoFeatureExtractor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        self._load()
        wav = self.pad_if_needed(wav, sr)
        audio_np = wav.squeeze().cpu().numpy()
        inputs = self._processor(
            audio_np, sampling_rate=sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze()
        return emb.cpu().numpy()
