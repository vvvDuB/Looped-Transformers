"""Local HTTP service contract: real experiments, inference, and bounded work."""
from contextlib import contextmanager
import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from looped_transformers.inference import ModelManager
from looped_transformers.server import create_app


@pytest.fixture
def service_files(tmp_path, monkeypatch):
    monkeypatch.delenv("LOOPED_TRANSFORMERS_API_KEY", raising=False)
    configs = tmp_path / "configs"
    configs.mkdir()
    data = tmp_path / "text.jsonl"
    data.write_text('{"id":"one","text":"Two plus two is four."}\n'
                    '{"id":"two","text":"A queue processes items in order."}\n', encoding="utf-8")
    config = {
        "name": "http-smoke",
        "model": {"backend": "toy", "num_layers": 4, "width": 8,
                  "num_experts": 3, "top_k": 2, "max_context": 64},
        "datasets": [{"path": "../text.jsonl", "kind": "text",
                      "split": "synthetic", "purpose": "smoke"}],
        "variants": [{"name": "loop2", "loop": {"start": 1, "end": 2, "steps": 2}}],
        "bootstrap_samples": 100,
        "output_dir": "../ignored-config-output",
    }
    (configs / "smoke.json").write_text(json.dumps(config), encoding="utf-8")
    return configs, tmp_path / "runs", config


@contextmanager
def client_for(service_files, manager=None):
    configs, output, _ = service_files
    with TestClient(create_app(configs, output, manager=manager)) as client:
        yield client


def await_job(client, job_id, timeout=10):
    deadline = time.monotonic() + timeout
    waiter = threading.Event()
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] in {"completed", "failed", "cancelled"}:
            return last
        waiter.wait(0.005)
    pytest.fail(f"Job failed to finish before deadline: {last}")


def test_real_job_lifecycle_persists_run_and_provenance(service_files):
    manager = ModelManager()
    with client_for(service_files, manager) as client:
        assert client.get("/api/health").json()["cache_policy"] == "full_context_recompute"
        assert client.get("/api/configs").json() == ["smoke.json"]
        assert client.get("/api/runs").json() == []
        submitted = client.post("/api/jobs", json={"config": "smoke.json"})
        assert submitted.status_code == 202
        job = await_job(client, submitted.json()["id"])
        assert job["status"] == "completed", job
        assert "cancel" not in job and "future" not in job
        runs = client.get("/api/runs").json()
        assert len(runs) == 1
        assert runs[0]["id"] == job["run_id"]
        assert runs[0]["status"] == "completed"
        response = client.get(f"/api/runs/{job['run_id']}")
        assert response.status_code == 200
        record = response.json()
        assert record["summary"]["run_id"] == job["run_id"]
        assert record["config"]["name"] == "http-smoke"
        assert record["config"]["output_dir"] == str(service_files[1].resolve())
        assert (service_files[1] / job["run_id"] / "summary.json").is_file()
        assert not (service_files[0].parent / "ignored-config-output").exists()
        assert client.get("/api/jobs").json()[0]["status"] == "completed"
    assert manager.status()["loaded"] is False


def test_generation_uses_named_variant_and_model_lifecycle(service_files):
    with client_for(service_files) as client:
        payload = {"config": "smoke.json", "prompt": "Hi", "variant": "loop2",
                   "max_new_tokens": 3, "temperature": 0.7, "seed": 12}
        first = client.post("/api/generate", json=payload)
        second = client.post("/api/generate", json=payload)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert first.json()["token_ids"] == second.json()["token_ids"]
        assert first.json()["inference_mode"] == "full_context_recompute"
        assert client.get("/api/models").json()["loaded"] is True
        assert client.post("/api/models/unload").json() == {"loaded": False}
        assert client.get("/api/models").json()["loaded"] is False
        assert client.post("/api/generate", json=payload | {"variant": "unknown"}).status_code == 422
        assert client.post("/api/generate", json=payload | {"prompt": "x" * 64}).status_code == 422
        assert client.post("/api/generate", json=payload | {"arbitrary": "value"}).status_code == 422


def test_config_traversal_absolute_paths_and_symlink_escape_are_rejected(service_files):
    configs, _, config = service_files
    outside = configs.parent / "outside.json"
    outside.write_text(json.dumps(config), encoding="utf-8")
    (configs / "escape.json").symlink_to(outside)
    with client_for(service_files) as client:
        assert "escape.json" not in client.get("/api/configs").json()
        for name in ("../outside.json", str(outside), "escape.json", "missing.json", "../text.jsonl"):
            result = client.post("/api/jobs", json={"config": name})
            assert result.status_code == 404, (name, result.text)
        assert client.get("/api/jobs").json() == []
        assert client.get("/api/jobs/missing").status_code == 404
        assert client.post("/api/jobs/missing/cancel").status_code == 404
        assert client.get("/api/runs/invalid").status_code == 404


def test_foreign_origin_writes_are_blocked_before_work_is_started(service_files):
    with client_for(service_files) as client:
        payload = {"config": "smoke.json", "prompt": "Hi", "max_new_tokens": 1}
        for endpoint, body in (("/api/jobs", {"config": "smoke.json"}),
                               ("/api/generate", payload), ("/api/models/unload", {})):
            response = client.post(endpoint, json=body, headers={"Origin": "https://foreign.example"})
            assert response.status_code == 403
        allowed = client.post("/api/generate", json=payload, headers={"Origin": "http://testserver"})
        assert allowed.status_code == 200, allowed.text
        assert client.get("/api/jobs").json() == []
        assert allowed.headers["x-content-type-options"] == "nosniff"
        assert allowed.headers["cache-control"] == "no-store"


def test_optional_bearer_key_guards_api_reads_and_writes(service_files, monkeypatch):
    monkeypatch.setenv("LOOPED_TRANSFORMERS_API_KEY", "local-test-key")
    with client_for(service_files) as client:
        assert client.get("/api/health").status_code == 401
        assert client.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/api/jobs", json={"config": "smoke.json"}).status_code == 401
        authenticated = client.get("/api/health", headers={"Authorization": "Bearer local-test-key"})
        assert authenticated.status_code == 200


def test_queue_is_serial_bounded_and_pending_jobs_can_be_cancelled(service_files, monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    calls = []

    def controlled_experiment(experiment, manager, progress):
        calls.append(experiment.name)
        first_started.set()
        if not release_first.wait(timeout=10):
            raise RuntimeError("test did not release worker")
        return {"run_id": "1" * 32}

    monkeypatch.setattr("looped_transformers.server.run_experiment", controlled_experiment)
    try:
        with client_for(service_files) as client:
            first = client.post("/api/jobs", json={"config": "smoke.json"})
            assert first.status_code == 202
            assert first_started.wait(timeout=5)
            pending = []
            for _ in range(7):
                response = client.post("/api/jobs", json={"config": "smoke.json"})
                assert response.status_code == 202
                pending.append(response.json()["id"])
            assert len(calls) == 1
            assert all(client.get(f"/api/jobs/{job_id}").json()["status"] == "queued" for job_id in pending)
            assert client.post("/api/jobs", json={"config": "smoke.json"}).status_code == 429
            for job_id in pending:
                assert client.post(f"/api/jobs/{job_id}/cancel").json()["cancel_requested"] is True
            release_first.set()
            assert await_job(client, first.json()["id"])["status"] == "completed"
            for job_id in pending:
                assert await_job(client, job_id)["status"] == "cancelled"
            assert len(calls) == 1
            # Cancelled queue entries release their capacity for future work.
            subsequent = client.post("/api/jobs", json={"config": "smoke.json"})
            assert subsequent.status_code == 202
            assert await_job(client, subsequent.json()["id"])["status"] == "completed"
            assert len(calls) == 2
    finally:
        release_first.set()


def test_studio_derived_recipe_is_valid_and_executable(service_files):
    with client_for(service_files) as client:
        detail = client.get("/api/configs/smoke.json")
        assert detail.status_code == 200
        assert detail.json()["model"]["backend"] == "toy"
        assert detail.json()["variants"][0]["name"] == "loop2"
        payload = {
            "config": "smoke.json", "name": "studio-smoke",
            "variant_name": "euler-loop", "start": 1, "end": 2,
            "steps": 2, "mode": "block", "strategy": "euler",
            "anchor": 0.5,
        }
        saved = client.post("/api/configs/derive", json=payload)
        assert saved.status_code == 201, saved.text
        name = saved.json()["config"]
        assert name == "generated/studio-smoke.json"
        assert name in client.get("/api/configs").json()
        cfg = client.get("/api/configs/" + name).json()
        assert cfg["variants"][0]["loop"]["steps"] == 2
        assert cfg["model"]["backend"] == "toy"
        assert client.post("/api/configs/derive", json=payload).status_code == 409
        queued = client.post("/api/jobs", json={"config": name})
        assert queued.status_code == 202
        assert await_job(client, queued.json()["id"])["status"] == "completed"


def test_studio_recipe_rejects_invalid_and_unsafe_input(service_files):
    with client_for(service_files) as client:
        recipe = {
            "config": "smoke.json", "name": "safe-name",
            "variant_name": "euler", "start": 1, "end": 2,
            "steps": 2, "mode": "block", "strategy": "euler",
            "anchor": 0.5,
        }
        for change in (
            {"name": "../traversal"}, {"variant_name": "baseline"},
            {"start": 3, "end": 1}, {"end": 99},
            {"steps": 0}, {"steps": 1000}, {"anchor": 1.2},
            {"config": "../outside.json"}, {"model": {"model_id": "not-allowed"}},
        ):
            response = client.post("/api/configs/derive", json=recipe | change)
            assert response.status_code in {404, 422}, (change, response.text)
        assert not (service_files[0] / "generated").exists()


def test_projects_model_selection_and_scoped_runs(service_files):
    with client_for(service_files) as client:
        initial = client.get("/api/projects").json()
        assert len(initial) == 1
        default_id = initial[0]["id"]
        assert initial[0]["config_count"] == 1
        created = client.post("/api/projects", json={"name": "Qwen loop research"})
        assert created.status_code == 201, created.text
        project_id = created.json()["id"]
        assert client.get(f"/api/projects/{project_id}/models").json() == []
        assert client.post("/api/jobs", json={
            "config": "smoke.json", "project_id": project_id}).status_code == 404
        linked = client.post(f"/api/projects/{project_id}/configs",
                             json={"config": "smoke.json"})
        assert linked.status_code == 201, linked.text
        models = client.get(f"/api/projects/{project_id}/models").json()
        assert len(models) == 1
        assert models[0]["configs"] == ["smoke.json"]
        submitted = client.post("/api/jobs", json={
            "config": "smoke.json", "project_id": project_id})
        assert submitted.status_code == 202, submitted.text
        done = await_job(client, submitted.json()["id"])
        assert done["status"] == "completed", done
        project_runs = client.get(f"/api/projects/{project_id}/runs").json()
        assert [run["id"] for run in project_runs] == [done["run_id"]]
        assert client.get(f"/api/projects/{default_id}/runs").json() == []
        recipe = client.post("/api/configs/derive", json={
            "config": "smoke.json", "project_id": project_id,
            "name": "project-loop", "variant_name": "euler2",
            "start": 1, "end": 2, "steps": 2,
            "mode": "block", "strategy": "euler", "anchor": 0.5})
        assert recipe.status_code == 201, recipe.text
        config_name = recipe.json()["config"]
        assert config_name in client.get(f"/api/projects/{project_id}").json()["configs"]
        assert config_name not in client.get(f"/api/projects/{default_id}").json()["configs"]
        assert client.post("/api/jobs", json={"config": config_name}).status_code == 404
        assert len(client.get(f"/api/projects/{project_id}/models").json()[0]["configs"]) == 2


def test_project_membership_persists_and_rejects_unknown_projects(service_files):
    configs, output, _ = service_files
    with client_for(service_files) as client:
        project_id = client.post("/api/projects", json={
            "name": "Persistent Research", "description": "An isolated workspace"}).json()["id"]
        assert client.post(f"/api/projects/{project_id}/configs", json={
            "config": "smoke.json"}).status_code == 201
        assert client.get("/api/projects/" + "0" * 32).status_code == 404
        assert client.post("/api/projects", json={"name": "   "}).status_code == 422
        assert client.post("/api/jobs", json={
            "config": "smoke.json", "project_id": "0" * 32}).status_code == 404
    with TestClient(create_app(configs, output)) as client:
        saved = client.get(f"/api/projects/{project_id}")
        assert saved.status_code == 200
        assert saved.json()["configs"] == ["smoke.json"]
        assert saved.json()["name"] == "Persistent Research"
