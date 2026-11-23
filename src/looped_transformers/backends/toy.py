"""A small, random-weight causal Transformer with real sparse top-k MoE layers.

This backend is an offline engineering fixture. It is deliberately NOT trained,
so its text, perplexity, and benchmark scores are not evidence of model quality.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any, Sequence

import numpy as np

from .base import ModelOutput, validate_ids
from ..config import LoopConfig, ModelConfig
from ..looping import refine


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-5))


class ToyMoEBackend:
    vocab_size = 258
    bos_token_id = 1
    eos_token_id = 0

    def __init__(self, config: ModelConfig):
        if config.backend != "toy":
            raise ValueError("ToyMoEBackend requires backend='toy'")
        self.config = config
        self.max_context = config.max_context
        self.num_layers = config.num_layers
        self.num_experts = config.num_experts
        rng = np.random.default_rng(config.seed)
        width = config.width
        def weight(*shape: int, scale: float = 1.0) -> np.ndarray:
            value = (rng.standard_normal(shape) * (scale / np.sqrt(shape[-2]))).astype(np.float32)
            value.setflags(write=False)
            return value
        self.embedding = weight(self.vocab_size, width, scale=2.0)
        self.head = weight(width, self.vocab_size)
        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                "q": weight(width, width), "k": weight(width, width),
                "v": weight(width, width), "o": weight(width, width),
                "router": weight(width, config.num_experts),
                "up": weight(config.num_experts, width, 2 * width),
                "down": weight(config.num_experts, 2 * width, width),
            })
        angles = np.arange(config.max_context, dtype=np.float32)[:, None] / np.power(
            np.float32(10000), np.arange(width, dtype=np.float32)[None, :] / width)
        self.position = (np.float32(0.02) * np.sin(angles)).astype(np.float32)
        self.position.setflags(write=False)
        digest = hashlib.sha256()
        for array in self._weights():
            digest.update(array.tobytes())
        self._fingerprint = digest.hexdigest()

    def _weights(self):
        yield self.embedding
        yield self.head
        yield self.position
        for layer in self.layers:
            yield from layer.values()

    def tokenize(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [byte + 2 for byte in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        values = list(ids)
        if any(not isinstance(x, (int, np.integer)) or x < 0 or x >= self.vocab_size for x in values):
            raise ValueError("invalid token id")
        return bytes(int(i) - 2 for i in values if i >= 2).decode("utf-8", errors="replace")

    def forward(self, ids: Sequence[int], loop: LoopConfig | None = None,
                trace: bool = True) -> ModelOutput:
        tokens = validate_ids(ids, self.vocab_size, self.max_context)
        if loop is not None:
            loop.validate(self.num_layers)
        h = self.embedding[tokens] + self.position[:len(tokens)]
        hidden: dict[str, np.ndarray] = {}
        routing: list[dict[str, Any]] = []
        visits = [0] * self.num_layers
        causal_mask = np.triu(np.ones((len(tokens), len(tokens)), dtype=bool), k=1)

        def apply_layer(index: int, x: np.ndarray) -> np.ndarray:
            visits[index] += 1
            weights = self.layers[index]
            z = _norm(x)
            q, k, v = z @ weights["q"], z @ weights["k"], z @ weights["v"]
            scores = (q @ k.T) / np.sqrt(np.float32(self.config.width))
            scores = np.where(causal_mask, np.float32(-np.inf), scores)
            x = x + np.float32(0.15) * ((_softmax(scores) @ v) @ weights["o"])
            z = _norm(x)
            router_logits = z @ weights["router"]
            # Stable sorting gives deterministic expert tie-breaking in this fixture.
            expert_ids = np.argsort(-router_logits, axis=-1, kind="stable")[:, :self.config.top_k]
            selected = np.take_along_axis(router_logits, expert_ids, axis=-1)
            probabilities = _softmax(selected)
            mixed = np.zeros_like(x)
            for expert in range(self.num_experts):
                rows, slots = np.where(expert_ids == expert)
                if rows.size:
                    state = np.tanh(z[rows] @ weights["up"][expert]) @ weights["down"][expert]
                    mixed[rows] += state * probabilities[rows, slots, None]
            x = x + np.float32(0.15) * mixed
            if trace:
                routing.append({"layer": index, "visit": visits[index],
                                "expert_ids": expert_ids.copy(),
                                "weights": probabilities.copy(),
                                "num_experts": self.num_experts,
                                "capture": "actual_selected_experts"})
                hidden[f"layer_{index}"] = x.copy()
            return x

        index = 0
        while index < self.num_layers:
            if loop is not None and loop.mode == "block" and index == loop.start:
                def window(x: np.ndarray) -> np.ndarray:
                    for layer_index in range(loop.start, loop.end + 1):
                        x = apply_layer(layer_index, x)
                    return x
                natural = window(h)
                h = refine(h, natural, window, loop)
                if trace:
                    hidden[f"layer_{loop.end}"] = h.copy()
                index = loop.end + 1
            else:
                natural = apply_layer(index, h)
                if loop is not None and loop.mode == "layer" and loop.start <= index <= loop.end:
                    h = refine(h, natural, lambda x: apply_layer(index, x), loop)
                    if trace:
                        hidden[f"layer_{index}"] = h.copy()
                else:
                    h = natural
                index += 1
        logits = (_norm(h) @ self.head).astype(np.float32)
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("non-finite logits from toy model")
        return ModelOutput(logits=logits, hidden_states=hidden, routing=routing,
                           metadata={"layer_visits": sum(visits), "visits_per_layer": visits,
                                     "cache_policy": "full_context_recompute",
                                     "hidden_capture": "last_visit; blended output at loop boundaries"})

    def reset_peak_memory_stats(self) -> None:
        """No GPU allocations exist in the NumPy fixture."""

    def info(self) -> dict[str, Any]:
        return {"backend": "toy", "model_id": self.config.model_id,
                "architecture": "numpy_causal_sparse_moe", "trained": False,
                "purpose": "offline engineering fixture; not a language-quality benchmark",
                "num_layers": self.num_layers, "num_experts": self.num_experts,
                "top_k": self.config.top_k, "width": self.config.width,
                "vocab_size": self.vocab_size, "max_context": self.max_context,
                "parameter_count": sum(x.size for x in self._weights()) - self.position.size,
                "weights_sha256": self._fingerprint, "seed": self.config.seed,
                "dtype": "float32", "device": "cpu", "numpy_version": np.__version__,
                "loop_support": "block/layer frozen-weight interventions",
                "routing_capture": "actual_selected_experts", "cache_policy": "full_context_recompute",
                "config": asdict(self.config)}
