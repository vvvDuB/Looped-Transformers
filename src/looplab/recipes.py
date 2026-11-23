"""Validated reloading of exported runtime interventions, without executing code.

A recipe identifies original weights plus an inference transformation. It cannot
claim to be a newly trained model or request arbitrary adapters / cache policies.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any

from . import __version__
from .backends.huggingface import require_adapter
from .config import LoopConfig, ModelConfig
from .experiment import Variant


_FIELDS = {
    "schema_version", "kind", "model", "resolved_model", "variant", "source_run",
    "requires", "weights_modified", "cache_policy", "note",
}


def _object(value: Any, name: str, allowed: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if allowed is not None and (extra := set(value) - allowed):
        raise ValueError(f"Unknown {name} fields: {sorted(extra)}")
    return value


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _no_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def load_recipe(path: str | Path) -> tuple[ModelConfig, LoopConfig]:
    """Load a version-one recipe and return configs; never load model weights here.

    If the run captured an immutable Hugging Face model revision, use that
    revision even when the original experiment requested a moving branch.
    Unknown structural fields and malformed capability assertions are rejected.
    The resolved_model object is informational metadata; only explicitly checked
    fields can affect loading, and arbitrary code/imports are never accepted.
    """
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Runtime recipe exceeds the 2 MiB limit")
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates,
                     parse_constant=_no_constant)
    raw = _object(raw, "recipe", _FIELDS)
    if missing := _FIELDS - set(raw):
        raise ValueError(f"Missing recipe fields: {sorted(missing)}")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Only recipe schema_version=1 is supported")
    if raw["kind"] != "looplab_runtime_recipe":
        raise ValueError("Unsupported recipe kind")
    if raw["requires"] != f"looplab=={__version__}":
        raise ValueError(f"Recipe requires {raw['requires']!r}; this loader supports looplab=={__version__}")
    if raw["weights_modified"] is not False:
        raise ValueError("Only recipes with weights_modified=false are supported")
    if raw["cache_policy"] != "full_context_recompute":
        raise ValueError("Only full_context_recompute recipes are supported")
    if not isinstance(raw["source_run"], str) or not raw["source_run"].strip():
        raise ValueError("source_run must be a nonempty string")
    if not isinstance(raw["note"], str):
        raise ValueError("note must be a string")
    model_data = _object(raw["model"], "model", set(ModelConfig.__dataclass_fields__))
    try:
        model = ModelConfig(**model_data)
    except TypeError as exc:
        raise ValueError(f"Invalid model configuration: {exc}") from exc
    variant = _object(raw["variant"], "variant", {"name", "loop"})
    if set(variant) != {"name", "loop"}:
        raise ValueError("variant must contain name and loop")
    loop_data = _object(variant["loop"], "loop", set(LoopConfig.__dataclass_fields__))
    try:
        loop = LoopConfig(**loop_data)
    except TypeError as exc:
        raise ValueError(f"Invalid loop configuration: {exc}") from exc
    Variant(variant["name"], loop)  # Reuse experiment-name validation.
    resolved = _object(raw["resolved_model"], "resolved_model")
    if resolved.get("backend") != model.backend:
        raise ValueError("resolved_model backend disagrees with the model configuration")
    if resolved.get("model_id") != model.model_id:
        raise ValueError("resolved_model model_id disagrees with the model configuration")
    num_layers = resolved.get("num_layers")
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("resolved_model.num_layers must be a positive integer")
    loop.validate(num_layers)
    if model.backend == "toy":
        if num_layers != model.num_layers:
            raise ValueError("Resolved toy layer count disagrees with the model configuration")
        if resolved.get("architecture") != "numpy_causal_sparse_moe":
            raise ValueError("Unsupported toy architecture in recipe")
    elif model.backend == "hf":
        architecture = resolved.get("architecture")
        if not isinstance(architecture, str):
            raise ValueError("resolved_model.architecture is required for Hugging Face recipes")
        require_adapter(architecture)
        revision = resolved.get("resolved_revision")
        if revision is not None:
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("resolved_revision must be a full 40-character Git commit SHA or null")
            model = replace(model, revision=revision)
    return model, loop
