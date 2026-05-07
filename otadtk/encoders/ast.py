"""AST encoder — d=768, Audio Spectrogram Transformer."""

import numpy as np
import torch
from ..model_loader import BaseEncoder
from ..model_loader import register_encoder


@register_encoder("ast")
class ASTEncoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "ast"
        self.target_sr = 16000
        self.embedding_dim = 768
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import ASTModel, ASTFeatureExtractor
        self._processor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self._model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
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
        inputs = self._processor(
            audio_np, sampling_rate=sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return emb.cpu().numpy().astype(np.float32)
