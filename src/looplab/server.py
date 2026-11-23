"""Local experiment dashboard and bounded job queue. No remote code/config submission."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
import hmac
import os
from pathlib import Path
import threading
from urllib.parse import urlparse
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from .experiment import load_experiment
from .inference import ModelManager, generate
from .runner import run_experiment
from .store import RunStore


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: str = Field(min_length=1, max_length=200)


class GenerateRequest(JobRequest):
    prompt: str = Field(max_length=100_000)
    variant: str = "baseline"
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0, le=5)
    seed: int = Field(default=42, ge=0)


def create_app(config_root="configs", output_root="runs", manager=None):
    config_root = Path(config_root).resolve()
    store = RunStore(output_root)
    manager = manager or ModelManager()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="looplab")
    slots = threading.BoundedSemaphore(8)
    state_lock = threading.Lock()
    jobs = {}
    api_key = os.environ.get("LOOPLAB_API_KEY")

    def resolve_config(name):
        path = (config_root / name).resolve()
        if not path.is_relative_to(config_root) or path.suffix not in {".json", ".yaml", ".yml"} or not path.is_file():
            raise HTTPException(status_code=404, detail="Unknown experiment config")
        try:
            return replace(load_experiment(path), output_dir=str(store.root))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @asynccontextmanager
    async def lifespan(app):
        yield
        with state_lock:
            for job in jobs.values():
                job["cancel"].set()
        executor.shutdown(wait=True, cancel_futures=True)
        manager.unload()

    app = FastAPI(title="LoopLab", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        from fastapi.responses import JSONResponse
        if request.url.path.startswith("/api/") and api_key:
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied, f"Bearer {api_key}"):
                return JSONResponse({"detail": "API key required"}, status_code=401)
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            parsed_origin = urlparse(origin)
            if (parsed_origin.netloc != request.headers.get("host")
                    or parsed_origin.scheme != request.url.scheme):
                return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "cache_policy": "full_context_recompute"}

    @app.get("/api/configs")
    def configs():
        return sorted(str(p.relative_to(config_root)) for p in config_root.rglob("*")
                      if p.suffix in {".json", ".yaml", ".yml"} and p.is_file() and p.resolve().is_relative_to(config_root))

    @app.get("/api/runs")
    def runs():
        return store.list()

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        try:
            return store.get(run_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/models")
    def models():
        return manager.status()

    @app.post("/api/models/unload")
    def unload():
        manager.unload()
        return {"loaded": False}

    def public_job(job):
        return {k: v for k, v in job.items() if k not in {"cancel", "future"}}

    @app.get("/api/jobs")
    def list_jobs():
        with state_lock:
            return [public_job(j) for j in jobs.values()]

    @app.post("/api/jobs", status_code=202)
    def submit(body: JobRequest):
        experiment = resolve_config(body.config)
        if not slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Experiment queue is full (8 jobs)")
        job_id = uuid.uuid4().hex
        event = threading.Event()
        with state_lock:
            # Keep bounded in-memory job history. Persistent experiment runs remain in SQLite.
            finished = [key for key, value in jobs.items() if value["status"] in {"completed", "failed", "cancelled"}]
            for key in finished[:-100]:
                jobs.pop(key)
            jobs[job_id] = {"id": job_id, "config": body.config, "status": "queued", "cancel": event}

        def work():
            try:
                if event.is_set():
                    with state_lock:
                        jobs[job_id]["status"] = "cancelled"
                    return
                with state_lock:
                    jobs[job_id]["status"] = "running"

                def progress(message):
                    if event.is_set():
                        raise InterruptedError("Experiment cancelled at an example boundary")
                    with state_lock:
                        jobs[job_id]["progress"] = message

                result = run_experiment(experiment, manager=manager, progress=progress)
                with state_lock:
                    jobs[job_id].update(status="completed", run_id=result["run_id"])
            except Exception as exc:
                with state_lock:
                    jobs[job_id].update(status="cancelled" if event.is_set() else "failed",
                                        error=f"{type(exc).__name__}: {exc}")
            finally:
                slots.release()

        future = executor.submit(work)
        with state_lock:
            jobs[job_id]["future"] = future
            return public_job(jobs[job_id])

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        with state_lock:
            if job_id not in jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            return public_job(jobs[job_id])

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        with state_lock:
            if job_id not in jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            if jobs[job_id]["status"] in {"completed", "failed", "cancelled"}:
                return public_job(jobs[job_id]) | {"cancel_requested": False}
            jobs[job_id]["cancel"].set()
            return public_job(jobs[job_id]) | {"cancel_requested": True}

    @app.post("/api/generate")
    def inference(body: GenerateRequest):
        experiment = resolve_config(body.config)
        variant = next((v for v in experiment.variants if v.name == body.variant), None)
        if body.variant != "baseline" and variant is None:
            raise HTTPException(status_code=422, detail="Unknown variant")
        try:
            with manager.lease(experiment.model) as backend:
                return generate(backend, body.prompt, variant.loop if variant else None,
                                body.max_new_tokens, body.temperature, body.seed)
        except (ValueError, RuntimeError, ImportError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
