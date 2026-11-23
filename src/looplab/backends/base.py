"""Backend contract; all outputs are CPU arrays, and no token cache is exposed."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from ..config import LoopConfig


@dataclass
class ModelOutput:
    logits: np.ndarray
    hidden_states: dict[str, np.ndarray] = field(default_factory=dict)
    routing: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    max_context: int
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | None

    def tokenize(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def forward(self, ids: Sequence[int], loop: LoopConfig | None = None,
                trace: bool = True) -> ModelOutput: ...
    def info(self) -> dict[str, Any]: ...
    def reset_peak_memory_stats(self) -> None: ...


def validate_ids(ids: Sequence[int], vocab_size: int, max_context: int) -> np.ndarray:
    values = np.asarray(ids)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("input_ids must be a nonempty one-dimensional sequence")
    if values.dtype.kind not in "iu":
        raise ValueError("input_ids must contain integers")
    if values.size > max_context:
        raise ValueError(f"input has {values.size} tokens; max_context is {max_context}")
    if np.any(values < 0) or np.any(values >= vocab_size):
        raise ValueError(f"input_ids must be between 0 and {vocab_size - 1}")
    return values.astype(np.int64, copy=False)
