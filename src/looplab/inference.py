"""One resident model, serialized mutations, and cache-free autoregressive inference."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import threading
import time

import numpy as np

from .backends import create_backend
from .config import LoopConfig, ModelConfig


class ModelManager:
    """Hold at most one model. Lease its complete experiment to prevent hook races."""
    def __init__(self):
        self._lock = threading.RLock()
        self._backend = None
        self._key = None

    @contextmanager
    def lease(self, config: ModelConfig):
        with self._lock:
            key = json.dumps(asdict(config), sort_keys=True)
            if key != self._key:
                self._backend = None
                self._key = None
                gc.collect()
                self._backend = create_backend(config)
                self._key = key
            yield self._backend

    def unload(self):
        with self._lock:
            self._backend = None
            self._key = None
            gc.collect()
            # Avoid importing torch for the offline backend.
            import sys
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def status(self) -> dict:
        with self._lock:
            return {"loaded": self._backend is not None,
                    "model": self._backend.info() if self._backend else None}


def generate(backend, prompt: str, loop: LoopConfig | None = None,
             max_new_tokens: int = 32, temperature: float = 0.0, seed: int = 42) -> dict:
    """Recompute the complete prefix each step; never masquerade as cached decode."""
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be between 1 and 4096")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    ids = list(backend.tokenize(prompt))
    if not ids:
        bos = getattr(backend, "bos_token_id", None)
        if bos is None:
            raise ValueError("Empty prompt and tokenizer has no BOS token")
        ids = [bos]
    if len(ids) >= backend.max_context:
        raise ValueError("Prompt fills the context window; shorten it or increase max_context")
    original_count = len(ids)
    rng = np.random.default_rng(seed)
    generated = []
    timings = []
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        if len(ids) >= backend.max_context:
            stop_reason = "context_limit"
            break
        started = time.perf_counter()
        output = backend.forward(ids, loop=loop, trace=False)
        logits = np.asarray(output.logits[-1], dtype=np.float64)
        if not np.all(np.isfinite(logits)):
            raise ValueError("Model produced non-finite logits")
        if temperature == 0:
            token = int(np.argmax(logits))
        else:
            scaled = (logits - np.max(logits)) / temperature
            probabilities = np.exp(scaled)
            probabilities /= probabilities.sum()
            token = int(rng.choice(len(probabilities), p=probabilities))
        timings.append(time.perf_counter() - started)
        ids.append(token)
        generated.append(token)
        if token == getattr(backend, "eos_token_id", None):
            stop_reason = "eos"
            break
    elapsed = sum(timings)
    return {"text": backend.decode(generated), "token_ids": generated,
            "prompt_tokens": original_count, "generated_tokens": len(generated),
            "stop_reason": stop_reason, "inference_mode": "full_context_recompute",
            "cache": False, "seed": seed, "temperature": temperature,
            "seconds": elapsed, "time_to_first_token_seconds": timings[0] if timings else None,
            "recompute_tokens_per_second": len(generated) / elapsed if elapsed else None,
            "per_token_seconds": timings, "model": backend.info()}
