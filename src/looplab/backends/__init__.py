"""Lazy backend imports keep the offline demo independent of PyTorch."""
from .base import Backend, ModelOutput
from ..config import ModelConfig


def create_backend(config: ModelConfig) -> Backend:
    if config.backend == "toy":
        from .toy import ToyMoEBackend
        return ToyMoEBackend(config)
    if config.backend == "hf":
        from .huggingface import HuggingFaceBackend
        return HuggingFaceBackend(config)
    raise ValueError(f"unknown backend {config.backend!r}")


__all__ = ["Backend", "ModelOutput", "create_backend"]
