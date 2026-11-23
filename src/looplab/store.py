"""Transactional local run registry and atomic JSON artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import uuid


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "registry.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, finished_at TEXT, config_json TEXT NOT NULL,
                summary_json TEXT, error TEXT)""")

    def connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("Invalid run ID")
        return self.root / run_id

    def start(self, name: str, config: dict) -> str:
        run_id = uuid.uuid4().hex
        folder = self.path(run_id)
        folder.mkdir()
        write_json(folder / "config.json", config)
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                       (run_id, name, "running", now(), None, json.dumps(config, allow_nan=False), None, None))
        return run_id

    def finish(self, run_id: str, summary: dict):
        write_json(self.path(run_id) / "summary.json", summary)
        with self.connect() as db:
            db.execute("UPDATE runs SET status='completed', finished_at=?, summary_json=? WHERE id=?",
                       (now(), json.dumps(summary, allow_nan=False), run_id))

    def fail(self, run_id: str, error: str, status: str = "failed"):
        if status not in {"failed", "cancelled"}:
            raise ValueError("Invalid terminal failure status")
        write_json(self.path(run_id) / "error.json", {"error": error, "at": now()})
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?, finished_at=?, error=? WHERE id=?",
                       (status, now(), error, run_id))

    def get(self, run_id: str) -> dict:
        self.path(run_id)
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        result["summary"] = json.loads(result.pop("summary_json") or "null")
        return result

    def list(self, limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute(
                "SELECT id,name,status,created_at,finished_at,error FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,))]

    def recover_interrupted(self) -> int:
        """Explicit operator recovery; never silently rewrite a live run."""
        with self.connect() as db:
            cursor = db.execute("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (now(),))
            return cursor.rowcount
