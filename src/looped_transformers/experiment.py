"""Strict, portable experiment descriptions. Paths are relative to the config file."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .config import LoopConfig, ModelConfig


def _keys(value: dict, allowed: set[str], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Unknown {context} fields: {sorted(extra)}")


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    kind: str = "text"
    limit: int | None = None
    split: str = "validation"
    purpose: str = "screening"

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("dataset.path must be a nonempty string")
        if self.kind not in {"text", "qa"}:
            raise ValueError("dataset.kind must be text or qa")
        if self.limit is not None and (type(self.limit) is not int or self.limit <= 0):
            raise ValueError("dataset.limit must be a positive integer or null")
        if self.split not in {"train", "validation", "test", "synthetic"}:
            raise ValueError("dataset.split must be train, validation, test, or synthetic")
        if self.purpose not in {"smoke", "screening", "confirmation"}:
            raise ValueError("dataset.purpose must be smoke, screening, or confirmation")
        if self.split == "test" and self.purpose == "screening":
            raise ValueError("Use validation for screening; reserve test for confirmation")


@dataclass(frozen=True)
class Variant:
    name: str
    loop: LoopConfig

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.name):
            raise ValueError("Variant names must be 1-64 letters, digits, underscores or hyphens")
        if self.name == "baseline":
            raise ValueError("baseline is reserved for the unmodified model")


@dataclass(frozen=True)
class Experiment:
    name: str
    model: ModelConfig
    datasets: tuple[DatasetSpec, ...]
    variants: tuple[Variant, ...]
    output_dir: str = "runs"
    stride: int | None = None
    bootstrap_samples: int = 1000
    seed: int = 42
    trace: bool = True
    source_path: str | None = field(default=None, compare=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Experiment name is required")
        if not self.datasets or not self.variants:
            raise ValueError("At least one dataset and one variant are required")
        if len({v.name for v in self.variants}) != len(self.variants):
            raise ValueError("Variant names must be unique")
        if len({(s.path, s.kind) for s in self.datasets}) != len(self.datasets):
            raise ValueError("Duplicate datasets in an experiment")
        if self.stride is not None and (type(self.stride) is not int or not 0 < self.stride < self.model.max_context):
            raise ValueError("stride must be positive and smaller than max_context")
        if type(self.bootstrap_samples) is not int or not 100 <= self.bootstrap_samples <= 100_000:
            raise ValueError("bootstrap_samples must be between 100 and 100000")
        if type(self.trace) is not bool:
            raise ValueError("trace must be boolean")

    def as_dict(self) -> dict:
        value = asdict(self)
        value.pop("source_path", None)
        return value


def experiment_from_dict(raw: dict[str, Any], base_dir: Path | None = None) -> Experiment:
    _keys(raw, {"name", "model", "datasets", "variants", "output_dir", "stride",
                "bootstrap_samples", "seed", "trace"}, "experiment")
    raw = dict(raw)
    model_data = raw.pop("model", {})
    _keys(model_data, set(ModelConfig.__dataclass_fields__), "model")
    model = ModelConfig(**model_data)
    specs = []
    for item in raw.pop("datasets", []):
        _keys(item, set(DatasetSpec.__dataclass_fields__), "dataset")
        item = dict(item)
        if base_dir is not None and "path" in item:
            item["path"] = str((base_dir / item["path"]).resolve())
        specs.append(DatasetSpec(**item))
    variants = []
    for item in raw.pop("variants", []):
        _keys(item, {"name", "loop"}, "variant")
        loop = item.get("loop", {})
        _keys(loop, set(LoopConfig.__dataclass_fields__), "loop")
        variants.append(Variant(name=item["name"], loop=LoopConfig(**loop)))
    if base_dir is not None:
        raw["output_dir"] = str((base_dir / raw.get("output_dir", "runs")).resolve())
    return Experiment(model=model, datasets=tuple(specs), variants=tuple(variants), **raw)


def load_experiment(path: str | Path) -> Experiment:
    from dataclasses import replace
    path = Path(path).resolve()
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(content)
    else:
        import yaml
        raw = yaml.safe_load(content)
    return replace(experiment_from_dict(raw, path.parent), source_path=str(path))
