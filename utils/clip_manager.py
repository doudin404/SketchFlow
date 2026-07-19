"""Shared OpenCLIP model loading and normalized feature encoding."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import open_clip
import torch


class CLIPManager:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
            weights_only=False,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images.to(self.device))
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@lru_cache(maxsize=4)
def get_clip_manager(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | None = None,
) -> CLIPManager:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return CLIPManager(model_name, pretrained, resolved_device)
