"""Strict local JSONL datasets with reproducible content and selection manifests.

The first nonempty line may be ``{"_meta": {"split": ..., "purpose": ...,
"source": ...}}``. It is metadata, never an example. Subsequent lines contain
text records or multiple-choice records. Nothing is downloaded implicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Dataset:
    rows: list[dict[str, Any]]
    fingerprint: str
    kind: str
    path: Path
    metadata: dict[str, Any]
    total_rows: int
    limit: int | None = None

    def manifest(self) -> dict[str, Any]:
        """Describe the exact source bytes and the evaluated, ordered selection."""
        selected = json.dumps(self.rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return {
            "path": str(self.path),
            "kind": self.kind,
            "sha256": self.fingerprint,
            "selection_sha256": hashlib.sha256(selected).hexdigest(),
            "examples": len(self.rows),
            "total_examples": self.total_rows,
            "limit": self.limit,
            "split": self.metadata.get("split", "unspecified"),
            "purpose": self.metadata.get("purpose", "user_supplied_evaluation"),
            "source": self.metadata.get("source", "local_jsonl"),
            "metadata": dict(self.metadata),
        }


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON number {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numeric value exceeds float64 limits")
    return number


def _nonempty_string(value: Any, name: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {name} must be a nonempty string")
    return value


def load_dataset(path: Path | str, kind: str = "text", limit: int | None = None) -> Dataset:
    """Load and validate a local text or multiple-choice JSONL dataset.

    Text rows: ``{"id": "optional-id", "text": "..."}``.
    QA rows: ``{"id": "optional-id", "question": "...", "choices": ["...", ...],
    "answer": 0}``, with a zero-based integer answer. Optional per-row ``metadata``
    must be an object. IDs omitted by the source are derived from physical line
    numbers. All source rows are validated, including those beyond ``limit``.

    ``fingerprint`` is SHA-256 of the *original file bytes*, including metadata
    and line endings; the manifest also hashes the selected normalized records.
    ``limit`` must be ``None`` or a positive integer. This avoids treating an
    accidental zero as an unlimited and potentially expensive evaluation.
    """
    if kind not in {"text", "qa"}:
        raise ValueError("kind must be 'text' or 'qa'")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0):
        raise ValueError("limit must be None or a positive integer")
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: dataset must be UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    metadata: dict[str, Any] = {}
    first_record_seen = False
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        context = f"{path.name}:{number}"
        try:
            row = json.loads(line, object_pairs_hook=_object_without_duplicates, parse_constant=_reject_constant, parse_float=_finite_json_float)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{context}: each record must be a JSON object")
        if "_meta" in row:
            if first_record_seen or set(row) != {"_meta"} or not isinstance(row["_meta"], dict):
                raise ValueError(f"{context}: _meta must be the sole field of the first record and contain an object")
            metadata = dict(row["_meta"])
            for key in ("split", "purpose", "source"):
                if key in metadata:
                    _nonempty_string(metadata[key], key, context)
            first_record_seen = True
            continue
        first_record_seen = True
        allowed = {"id", "metadata", "text"} if kind == "text" else {"id", "metadata", "question", "choices", "answer"}
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"{context}: unknown fields {sorted(unknown)!r}")
        identifier = _nonempty_string(row.get("id", f"line-{number}"), "id", context)
        if identifier in identifiers:
            raise ValueError(f"{context}: duplicate example id {identifier!r}")
        identifiers.add(identifier)
        record = dict(row)
        record["id"] = identifier
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError(f"{context}: metadata must be an object")
        if kind == "text":
            _nonempty_string(row.get("text"), "text", context)
        else:
            _nonempty_string(row.get("question"), "question", context)
            choices = row.get("choices")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
                raise ValueError(f"{context}: choices must be a list of 2 to 26 strings")
            for index, choice in enumerate(choices):
                _nonempty_string(choice, f"choices[{index}]", context)
            if len(set(choices)) != len(choices):
                raise ValueError(f"{context}: duplicate choices are not allowed")
            answer = row.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices):
                raise ValueError(f"{context}: answer must be a zero-based integer indexing choices")
        rows.append(record)
    if not rows:
        raise ValueError(f"{path}: dataset contains no examples")
    return Dataset(
        rows=rows if limit is None else rows[:int(limit)],
        fingerprint=hashlib.sha256(raw).hexdigest(),
        kind=kind,
        path=path,
        metadata=metadata,
        total_rows=len(rows),
        limit=None if limit is None else int(limit),
    )
