"""Task 2.2 - Multimodal vectorization engine.

Both functions return a plain list[float] of length 512, L2-normalised,
so cosine similarity == dot product (matches pgvector's cosine ops).
"""
from __future__ import annotations

import io
from typing import List, Optional

import torch
from PIL import Image

from .model_wrapper import CLIPModelWrapper, get_clip_model

EMBEDDING_DIM = 512


def _to_tensor(out) -> torch.Tensor:
    # Newer transformers versions may return a model-output object instead of a tensor.
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def _finalize(features: torch.Tensor) -> List[float]:
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    vec = features[0].detach().cpu().tolist()
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-D vector, got {len(vec)}")
    return vec


def generate_image_embedding(
    image_bytes: bytes, wrapper: Optional[CLIPModelWrapper] = None
) -> List[float]:
    wrapper = wrapper or get_clip_model()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid or unsupported image data") from exc

    inputs = wrapper.processor(images=image, return_tensors="pt").to(wrapper.device)
    with torch.no_grad():
        features = _to_tensor(wrapper.model.get_image_features(**inputs))
    return _finalize(features)


def generate_text_embedding(
    text_description: str, wrapper: Optional[CLIPModelWrapper] = None
) -> List[float]:
    if not text_description or not text_description.strip():
        raise ValueError("text_description must be a non-empty string")
    wrapper = wrapper or get_clip_model()

    inputs = wrapper.processor(
        text=[text_description],
        return_tensors="pt",
        padding=True,
        truncation=True,  # CLIP context length is 77 tokens
        max_length=77,
    ).to(wrapper.device)
    with torch.no_grad():
        features = _to_tensor(wrapper.model.get_text_features(**inputs))
    return _finalize(features)
