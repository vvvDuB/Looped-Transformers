"""Offline end-to-end inference and model lifecycle checks."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading

import numpy as np
import pytest

from looplab.backends import create_backend
from looplab.config import LoopConfig, ModelConfig
from looplab.inference import ModelManager, generate


def small_config(**changes):
    return ModelConfig(num_layers=4, width=8, num_experts=3, top_k=2,
                       max_context=32, **changes)


def test_greedy_and_seeded_sampling_are_reproducible():
    backend = create_backend(small_config())
    for temperature in (0.0, 0.8):
        first = generate(backend, "Hi", max_new_tokens=5, temperature=temperature, seed=81)
        second = generate(backend, "Hi", max_new_tokens=5, temperature=temperature, seed=81)
        assert first["token_ids"] == second["token_ids"]
        assert first["text"] == second["text"]
        assert first["cache"] is False
        assert first["inference_mode"] == "full_context_recompute"
        assert first["generated_tokens"] == len(first["token_ids"])
        assert len(first["per_token_seconds"]) == first["generated_tokens"]
        assert first["time_to_first_token_seconds"] >= 0
        json.dumps(first, allow_nan=False)


def test_generation_recomputes_every_complete_prefix(monkeypatch):
    backend = create_backend(small_config())
    monkeypatch.setattr(backend, "eos_token_id", None)
    original = backend.forward
    calls = []

    def record(ids, loop=None, trace=True):
        calls.append((list(ids), loop, trace))
        return original(ids, loop=loop, trace=trace)

    monkeypatch.setattr(backend, "forward", record)
    loop = LoopConfig(start=1, end=2, steps=2)
    result = generate(backend, "Hi", loop=loop, max_new_tokens=4)
    initial = backend.tokenize("Hi")
    assert len(calls) == 4
    for index, (ids, applied_loop, trace) in enumerate(calls):
        assert ids == initial + result["token_ids"][:index]
        assert applied_loop == loop
        assert trace is False
    assert result["stop_reason"] == "max_new_tokens"


def test_k_one_generation_reproduces_unmodified_model():
    backend = create_backend(small_config())
    baseline = generate(backend, "Hi", max_new_tokens=5, temperature=0.7, seed=20)
    equivalent = generate(backend, "Hi", loop=LoopConfig(1, 2, steps=1),
                          max_new_tokens=5, temperature=0.7, seed=20)
    restored = generate(backend, "Hi", max_new_tokens=5, temperature=0.7, seed=20)
    assert baseline["token_ids"] == equivalent["token_ids"] == restored["token_ids"]


def test_context_limit_is_reported_without_silent_prompt_truncation(monkeypatch):
    backend = create_backend(replace(small_config(), max_context=8))
    monkeypatch.setattr(backend, "eos_token_id", None)
    result = generate(backend, "1234567", max_new_tokens=5)
    assert result["prompt_tokens"] == 7
    assert result["generated_tokens"] == 1
    assert result["stop_reason"] == "context_limit"
    with pytest.raises(ValueError, match="Prompt fills"):
        generate(backend, "12345678", max_new_tokens=1)


def test_empty_prompt_uses_bos_and_eos_stops_generation(monkeypatch):
    backend = create_backend(small_config())
    original = backend.forward
    calls = []

    def eos_output(ids, loop=None, trace=True):
        calls.append(list(ids))
        output = original(ids, loop=loop, trace=trace)
        output.logits[-1] = np.full_like(output.logits[-1], -10.0)
        output.logits[-1, backend.eos_token_id] = 10.0
        return output

    monkeypatch.setattr(backend, "forward", eos_output)
    result = generate(backend, "", max_new_tokens=5)
    assert calls == [[backend.bos_token_id]]
    assert result["token_ids"] == [backend.eos_token_id]
    assert result["stop_reason"] == "eos"
    monkeypatch.setattr(backend, "bos_token_id", None)
    with pytest.raises(ValueError, match="no BOS"):
        generate(backend, "")


@pytest.mark.parametrize("options", [
    {"prompt": None}, {"max_new_tokens": 0}, {"max_new_tokens": True},
    {"max_new_tokens": 4097}, {"temperature": -1.0},
    {"temperature": float("nan")}, {"temperature": float("inf")},
])
def test_invalid_generation_parameters_are_rejected(options):
    backend = create_backend(small_config())
    parameters = {"prompt": "Hi", "max_new_tokens": 1} | options
    with pytest.raises(ValueError):
        generate(backend, **parameters)


def test_model_manager_reuses_one_model_and_unloads():
    manager = ModelManager()
    config = small_config()
    assert manager.status() == {"loaded": False, "model": None}
    with manager.lease(config) as first:
        fingerprint = first.info()["weights_sha256"]
    with manager.lease(config) as second:
        assert first is second
    with manager.lease(replace(config, seed=config.seed + 1)) as third:
        assert third is not first
        assert third.info()["weights_sha256"] != fingerprint
    assert manager.status()["loaded"] is True
    manager.unload()
    assert manager.status() == {"loaded": False, "model": None}


def test_model_leases_serialize_inference_across_threads():
    manager = ModelManager()
    config = small_config()
    attempting = threading.Event()
    entered = threading.Event()

    def acquire_second_lease():
        attempting.set()
        with manager.lease(config):
            entered.set()
        return True

    with ThreadPoolExecutor(max_workers=1) as executor:
        with manager.lease(config):
            future = executor.submit(acquire_second_lease)
            assert attempting.wait(timeout=5)
            assert not entered.wait(timeout=0.02)
        assert future.result(timeout=5)
    assert entered.is_set()
    manager.unload()
