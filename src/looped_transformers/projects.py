"""Persistent project -> model preset -> experiment/run associations.

Projects are local workspaces, not authentication or authorization boundaries.
The run registry remains the canonical storage for experiment artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import sqlite3
import uuid

from .store import RunStore, now


def model_key(model: dict) -> str:
    """Stable identity of a checkpoint, independent of loop recipes or evaluation datasets."""
    identity = {field: model.get(field) for field in ("backend", "model_id", "revision")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]


class ProjectStore:
    def __init__(self, run_store: RunStore):
        self.runs = run_store
        with self.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, is_default INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS project_configs (
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                config TEXT NOT NULL, added_at TEXT NOT NULL,
                PRIMARY KEY (project_id, config))""")
            db.execute("""CREATE TABLE IF NOT EXISTS project_runs (
                run_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                config TEXT NOT NULL, model_key TEXT NOT NULL, assigned_at TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_project_runs_project ON project_runs(project_id)")

    @contextmanager
    def connection(self):
        db = self.runs.connect()
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def bootstrap(self, available_configs: list[str]) -> str:
        """Only once, migrate historical runs/presets into an initial workspace."""
        with self.connection() as db:
            row = db.execute("SELECT id FROM projects WHERE is_default=1 LIMIT 1").fetchone()
            if row:
                return row[0]
            project_id = uuid.uuid4().hex
            timestamp = now()
            db.execute("INSERT INTO projects VALUES (?,?,?,?,?,1)",
                       (project_id, "General Research",
                        "Initial workspace containing existing model presets and legacy runs.",
                        timestamp, timestamp))
            db.executemany("INSERT INTO project_configs VALUES (?,?,?)",
                           ((project_id, name, timestamp) for name in available_configs))
            for run_id, config_raw in db.execute("SELECT id,config_json FROM runs"):
                config = json.loads(config_raw)
                db.execute("INSERT OR IGNORE INTO project_runs VALUES (?,?,?,?,?)",
                           (run_id, project_id, "", model_key(config.get("model", {})), timestamp))
        return project_id

    def default_id(self) -> str:
        with self.connection() as db:
            row = db.execute("SELECT id FROM projects WHERE is_default=1 LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("Project registry was not initialized")
        return row[0]

    def list(self) -> list[dict]:
        with self.connection() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("""SELECT p.*,
                (SELECT COUNT(*) FROM project_configs c WHERE c.project_id=p.id) AS config_count,
                (SELECT COUNT(*) FROM project_runs r WHERE r.project_id=p.id) AS run_count
                FROM projects p ORDER BY p.is_default DESC,p.created_at ASC""").fetchall()
            return [dict(row) for row in rows]

    def get(self, project_id: str) -> dict:
        with self.connection() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
            if not row:
                raise KeyError(project_id)
            configs = [r[0] for r in db.execute(
                "SELECT config FROM project_configs WHERE project_id=? ORDER BY added_at,config", (project_id,))]
            count = db.execute("SELECT COUNT(*) FROM project_runs WHERE project_id=?",
                               (project_id,)).fetchone()[0]
            return dict(row) | {"configs": configs, "config_count": len(configs), "run_count": count}

    def create(self, name: str, description: str = "") -> dict:
        name = name.strip()
        if not name:
            raise ValueError("Project name must not be blank")
        project_id = uuid.uuid4().hex
        timestamp = now()
        with self.connection() as db:
            db.execute("INSERT INTO projects VALUES (?,?,?,?,?,0)",
                       (project_id, name, description.strip(), timestamp, timestamp))
        return self.get(project_id)

    def attach(self, project_id: str, config: str) -> None:
        self.get(project_id)
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO project_configs VALUES (?,?,?)",
                       (project_id, config, now()))
            db.execute("UPDATE projects SET updated_at=? WHERE id=?", (now(), project_id))

    def detach(self, project_id: str, config: str) -> None:
        self.get(project_id)
        with self.connection() as db:
            db.execute("DELETE FROM project_configs WHERE project_id=? AND config=?",
                       (project_id, config))
            db.execute("UPDATE projects SET updated_at=? WHERE id=?", (now(), project_id))

    def contains(self, project_id: str, config: str) -> bool:
        self.get(project_id)
        with self.connection() as db:
            return db.execute("SELECT 1 FROM project_configs WHERE project_id=? AND config=?",
                              (project_id, config)).fetchone() is not None

    def record_run(self, project_id: str, run_id: str, config: str, model_identity: str):
        self.get(project_id)
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO project_runs VALUES (?,?,?,?,?)",
                       (run_id, project_id, config, model_identity, now()))

    def list_runs(self, project_id: str, limit: int = 1000) -> list[dict]:
        self.get(project_id)
        with self.connection() as db:
            db.row_factory = sqlite3.Row
            result = db.execute("""SELECT r.id,r.name,r.status,r.created_at,r.finished_at,r.error,
                p.config,p.model_key FROM project_runs p JOIN runs r ON r.id=p.run_id
                WHERE p.project_id=? ORDER BY r.created_at DESC LIMIT ?""", (project_id, limit)).fetchall()
            return [dict(row) for row in result]
