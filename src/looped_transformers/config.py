"""Strict configuration objects shared by backends, runners, and the service."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


def _integer(value: Any, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class LoopConfig:
    """A reversible inference transformation; layer bounds are inclusive, zero based."""

    start: int
    end: int
    steps: int = 2
    mode: str = "block"
    strategy: str = "euler"
    anchor: float = 0.0

    def __post_init__(self) -> None:
        _integer(self.start, "start", 0)
        _integer(self.end, "end", 0)
        _integer(self.steps, "steps", 1)
        if self.start > self.end:
            raise ValueError("start must be <= end")
        if self.mode not in {"block", "layer"}:
            raise ValueError("mode must be 'block' or 'layer'")
        if self.strategy not in {"euler", "naive"}:
            raise ValueError("strategy must be 'euler' or 'naive'")
        if isinstance(self.anchor, bool) or not isinstance(self.anchor, (int, float)):
            raise ValueError("anchor must be a finite number in [0, 1]")
        if not math.isfinite(self.anchor) or not 0 <= self.anchor <= 1:
            raise ValueError("anchor must be a finite number in [0, 1]")

    def validate(self, num_layers: int) -> None:
        _integer(num_layers, "num_layers", 1)
        if self.end >= num_layers:
            raise ValueError(f"loop window [{self.start}, {self.end}] exceeds {num_layers} layers")

    @property
    def extra_layer_visits(self) -> int:
        return (self.end - self.start + 1) * (self.steps - 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "toy"
    model_id: str = "toy-moe"
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    max_context: int = 128
    seed: int = 42
    width: int = 32
    num_layers: int = 6
    num_experts: int = 4
    top_k: int = 2
    load_in_4bit: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"toy", "hf"}:
            raise ValueError("backend must be 'toy' or 'hf'")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("revision must be a nonempty string or null")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        for name in ("max_context", "width", "num_layers", "num_experts", "top_k"):
            _integer(getattr(self, name), name, 1)
        if self.max_context < 2:
            raise ValueError("max_context must be >= 2")
        _integer(self.seed, "seed", 0)
        if self.top_k > self.num_experts:
            raise ValueError("top_k must not exceed num_experts")
        if type(self.load_in_4bit) is not bool or type(self.local_files_only) is not bool:
            raise ValueError("load_in_4bit and local_files_only must be booleans")
        if self.backend == "toy" and (self.device != "cpu" or self.load_in_4bit):
            raise ValueError("the toy backend uses NumPy on CPU; quantization is HF-only")
        if self.backend == "toy" and self.dtype != "float32":
            raise ValueError("the toy backend uses float32")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
