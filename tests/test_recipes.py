"""Offline recipe roundtrips and rejection of ambiguous or unsupported assertions."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from looplab import __version__
from looplab.backends import create_backend
from looplab.config import LoopConfig, ModelConfig
from looplab.recipes import load_recipe


def _recipe(model=None, resolved=None):
    model = model or ModelConfig(width=8, num_layers=4, max_context=32)
    if resolved is None:
        resolved = create_backend(model).info()
    return {
        "schema_version": 1, "kind": "looplab_runtime_recipe", "model": asdict(model),
        "resolved_model": resolved,
        "variant": {"name": "layer_k2", "loop": asdict(LoopConfig(1, 2, mode="layer"))},
        "source_run": "20261123-fixture", "requires": f"looplab=={__version__}",
        "weights_modified": False, "cache_policy": "full_context_recompute",
        "note": "Reload original weights and apply the runtime recipe.",
    }


def _write(tmp_path, data):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_toy_roundtrip_reproduces_the_exact_transformed_logits(tmp_path):
    recipe = _recipe()
    model, loop = load_recipe(_write(tmp_path, recipe))
    assert asdict(model) == recipe["model"]
    assert asdict(loop) == recipe["variant"]["loop"]
    original = create_backend(ModelConfig(**recipe["model"]))
    restored = create_backend(model)
    ids = original.tokenize("Test recipe")
    np.testing.assert_array_equal(original.forward(ids, LoopConfig(**recipe["variant"]["loop"])).logits,
                                  restored.forward(ids, loop).logits)


def test_hf_recipe_pins_resolved_commit_without_loading_or_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recipe loading must not load weights")
    monkeypatch.setattr("looplab.backends.create_backend", forbidden)
    model = ModelConfig(backend="hf", model_id="Qwen/Qwen3-0.6B-Base", revision="main")
    sha = "a" * 40
    resolved = {"backend": "hf", "model_id": model.model_id, "architecture": "qwen3",
                "num_layers": 28, "resolved_revision": sha}
    actual, loop = load_recipe(_write(tmp_path, _recipe(model, resolved)))
    assert actual.revision == sha
    assert actual.model_id == model.model_id
    assert loop.start == 1


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("schema_version", 2), ("kind", "pickle"),
    ("weights_modified", True), ("weights_modified", 0),
    ("cache_policy", "shared_kv"), ("requires", "looplab==99.0.0"),
    ("source_run", ""), ("note", []), ("variant", {}), ("resolved_model", []),
])
def test_recipe_rejects_invalid_capability_or_schema(tmp_path, key, value):
    recipe = _recipe()
    recipe[key] = value
    with pytest.raises(ValueError):
        load_recipe(_write(tmp_path, recipe))


@pytest.mark.parametrize("path,field,value", [
    ((), "script", "arbitrary.py"), (("model",), "trust_remote_code", True),
    (("variant", "loop"), "cache", True), (("model",), "backend", "remote"),
    (("resolved_model",), "model_id", "different"),
    (("resolved_model",), "num_layers", 2),
])
def test_recipe_rejects_unknown_fields_and_inconsistent_metadata(tmp_path, path, field, value):
    recipe = _recipe()
    target = recipe
    for name in path:
        target = target[name]
    target[field] = value
    with pytest.raises(ValueError):
        load_recipe(_write(tmp_path, recipe))


def test_recipe_rejects_native_loop_adapter_and_bad_revision(tmp_path):
    model = ModelConfig(backend="hf", model_id="example/checkpoint")
    for architecture, revision in (("inkling2_loop", "a" * 40), ("qwen3", "main")):
        resolved = {"backend": "hf", "model_id": model.model_id, "architecture": architecture,
                    "num_layers": 24, "resolved_revision": revision}
        with pytest.raises(ValueError):
            load_recipe(_write(tmp_path, _recipe(model, resolved)))


def test_recipe_rejects_missing_fields_duplicate_keys_and_nonfinite_json(tmp_path):
    recipe = _recipe()
    del recipe["weights_modified"]
    with pytest.raises(ValueError):
        load_recipe(_write(tmp_path, recipe))
    for text in ('{"kind": "one", "kind": "two"}', '{"anchor": NaN}'):
        path = tmp_path / "invalid.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_recipe(path)
