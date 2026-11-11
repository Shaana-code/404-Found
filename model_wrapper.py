"""Task 2.1 - Model management wrapper.

Loads openai/clip-vit-base-patch32 exactly once per process and handles
device placement (CUDA -> Apple MPS -> CPU).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import torch
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/clip-vit-base-patch32"


def _select_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CLIPModelWrapper:
    """Holds the CLIP model + processor. Use get_clip_model() to get the shared instance."""

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = _select_device(device)
        logger.info("Loading %s on %s ...", model_name, self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # inference only
        logger.info("CLIP model ready.")

    @property
    def embedding_dim(self) -> int:
        return self.model.config.projection_dim  # 512 for ViT-B/32


# ---- process-wide singleton -------------------------------------------------
_instance: Optional[CLIPModelWrapper] = None
_lock = threading.Lock()


def get_clip_model() -> CLIPModelWrapper:
    """Thread-safe lazy singleton. Call once in FastAPI startup to warm it up."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CLIPModelWrapper()
    return _instance
