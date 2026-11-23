"""End-to-end scientific accounting checks; all fixtures are local and untrained."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from looped_transformers.backends.toy import ToyMoEBackend
from looped_transformers.config import LoopConfig, ModelConfig
from looped_transformers.experiment import DatasetSpec, Experiment, Variant, experiment_from_dict, load_experiment
from looped_transformers.inference import ModelManager
from looped_transformers.runner import run_experiment, text_windows
from looped_transformers.store import RunStore


@pytest.mark.parametrize("length,context,stride", [
    (2, 2, 1), (3, 2, 1), (10, 4, 1), (10, 4, 3), (21, 8, 4),
    (22, 8, 7), (8, 16, 7), (65, 16, 8), (128, 32, 31),
])
def test_windows_score_each_target_once_with_a_causal_predecessor(length, context, stride):
    # Token IDs identify their original document positions; no model is involved.
    original = list(range(length))
    observed_targets = []
    for tokens, mask in text_windows(original, context, stride):
        assert 2 <= len(tokens) <= context
        assert mask.shape == (len(tokens) - 1,)
        assert mask.dtype == np.bool_
        assert mask.any()
        for local_prediction_position in np.flatnonzero(mask):
            target = tokens[local_prediction_position + 1]
            predecessor = tokens[local_prediction_position]
            assert predecessor == target - 1
            assert tokens[:local_prediction_position + 1] == original[tokens[0]:target]
            observed_targets.append(target)
    assert observed_targets == original[1:]


@pytest.mark.parametrize("context,stride", [(1, 1), (4, 0), (4, 4), (4, 5), (4, -1)])
def test_windows_reject_missing_overlap(context, stride):
    with pytest.raises(ValueError, match="stride"):
        list(text_windows([0, 1, 2], context, stride))


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _model_config():
    return ModelConfig(max_context=16, width=8, num_layers=4, num_experts=3, top_k=1)


def _experiment(tmp_path, rows=None, **overrides):
    path = _write_jsonl(tmp_path / "corpus.jsonl", rows or [
        {"id": "short", "text": "A short test."},
        {"id": "long", "text": "A longer text makes multiple overlapping windows necessary."},
    ])
    config = dict(name="accounting", model=_model_config(),
                  datasets=(DatasetSpec(str(path), split="synthetic", purpose="smoke"),),
                  variants=(Variant("identity", LoopConfig(1, 2, steps=1)),
                            Variant("euler", LoopConfig(1, 2, steps=2, mode="layer"))),
                  output_dir=str(tmp_path / "runs"), stride=7, bootstrap_samples=100)
    config.update(overrides)
    return Experiment(**config)


def test_toy_run_preserves_identity_token_counts_provenance_and_baseline(tmp_path):
    experiment = _experiment(tmp_path)
    manager = ModelManager()
    with manager.lease(experiment.model) as backend:
        ids = backend.tokenize("A probe.")
        original = backend.forward(ids, trace=False).logits.copy()
        fingerprint = backend.info()["weights_sha256"]
    summary = run_experiment(experiment, manager=manager)
    assert summary["status"] == "completed"
    assert summary["model"]["trained"] is False
    assert summary["checks"]["k1_identity"] is True
    assert summary["checks"]["baseline_restored"] is True
    with manager.lease(experiment.model) as backend:
        np.testing.assert_array_equal(backend.forward(ids, trace=False).logits, original)
        assert backend.info()["weights_sha256"] == fingerprint
    store = RunStore(experiment.output_dir)
    run = store.get(summary["run_id"])
    assert run["status"] == "completed"
    folder = store.path(summary["run_id"])
    outcome = json.loads((folder / "dataset-0.json").read_text())
    baseline = outcome["metrics"]["baseline"]
    assert baseline == outcome["metrics"]["identity"]
    expected_count = sum(len(row["text"].encode("utf-8")) - 1 for row in [
        json.loads(line) for line in (tmp_path / "corpus.jsonl").read_text().splitlines()
    ])
    assert baseline["token_count"] == expected_count
    document_results = [row["metrics"]["baseline"] for row in outcome["per_item"]]
    expected_nll = sum(row["nll_sum"] for row in document_results)
    assert baseline["mean_nll"] == pytest.approx(expected_nll / expected_count)
    assert baseline["perplexity"] == pytest.approx(np.exp(expected_nll / expected_count))
    proximity = outcome["comparisons"]["identity"]["logits"]
    assert proximity["token_count"] == expected_count
    assert proximity["top1_agreement"] == 1.0
    assert proximity["kl_baseline_to_candidate"] == pytest.approx(0.0, abs=1e-12)
    for metric in outcome["comparisons"]["identity"]["hidden_states"].values():
        assert metric["relative_l2"] == pytest.approx(0.0)
        assert metric["cosine_similarity"] == pytest.approx(1.0)
    for name in ("identity", "euler"):
        comparison = outcome["comparisons"][name]
        interval = comparison["token_weighted_mean_nll_paired_ci"]
        assert interval["weighted"] is True
        assert interval["delta"] == pytest.approx(comparison["mean_nll_delta"], abs=1e-12)
    assert outcome["comparisons"]["identity"]["routing_by_window"]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["datasets"][0]["sha256"] == hashlib.sha256((tmp_path / "corpus.jsonl").read_bytes()).hexdigest()
    assert manifest["datasets"][0]["selection_sha256"]
    assert manifest["datasets"][0]["purpose"] == "smoke"
    assert manifest["model"]["weights_sha256"] == fingerprint
    manager.unload()


def test_untraced_run_does_not_invent_routing_or_hidden_measurements(tmp_path):
    result = run_experiment(_experiment(tmp_path, trace=False))
    comparison = result["datasets"][0]["comparisons"]["euler"]
    assert comparison["hidden_states"] == {}
    outcome = json.loads((RunStore(tmp_path / "runs").path(result["run_id"]) / "dataset-0.json").read_text())
    for routing in outcome["comparisons"]["euler"]["routing_by_window"]:
        assert routing["available"] is False
        assert routing["mean_topk_jaccard"] is None


class _FixedManager:
    def __init__(self, backend):
        self.backend = backend

    @contextmanager
    def lease(self, config):
        yield self.backend


def test_dense_backend_without_routers_is_a_valid_text_experiment(tmp_path):
    experiment = _experiment(tmp_path, trace=False)
    backend = ToyMoEBackend(experiment.model)
    info = backend.info
    # The runner must handle a dense backend's documented zero-expert count.
    backend.info = lambda: info() | {"num_experts": 0, "routing_capture": "dense model"}
    result = run_experiment(experiment, manager=_FixedManager(backend))
    assert result["status"] == "completed"
    assert result["datasets"][0]["comparisons"]["identity"]["logits"]["top1_agreement"] == 1.0


def test_failed_identity_is_persisted_as_failed_run(tmp_path):
    experiment = _experiment(tmp_path)
    backend = ToyMoEBackend(experiment.model)
    forward = backend.forward

    def broken_identity(ids, loop=None, trace=True):
        output = forward(ids, loop=loop, trace=trace)
        if loop is not None and loop.steps == 1:
            output.logits += 0.1
        return output

    backend.forward = broken_identity
    with pytest.raises(RuntimeError, match="K=1 identity failed"):
        run_experiment(experiment, manager=_FixedManager(backend))
    rows = RunStore(experiment.output_dir).list()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "K=1 identity failed" in rows[0]["error"]


def test_failed_restoration_is_not_reported_as_success(tmp_path):
    experiment = _experiment(tmp_path)
    backend = ToyMoEBackend(experiment.model)
    forward = backend.forward
    mutated = False

    def broken_restore(ids, loop=None, trace=True):
        nonlocal mutated
        output = forward(ids, loop=loop, trace=trace)
        if loop is not None and loop.steps > 1:
            mutated = True
        if mutated:
            output.logits += 0.1
        return output

    backend.forward = broken_restore
    with pytest.raises(RuntimeError, match="Baseline restoration failed"):
        run_experiment(experiment, manager=_FixedManager(backend))
    assert RunStore(experiment.output_dir).list()[0]["status"] == "failed"


def test_duplicate_holdout_content_cannot_hide_behind_metadata(tmp_path):
    experiment = _experiment(tmp_path)
    screen = _write_jsonl(tmp_path / "screen.jsonl", [{"id": "s", "text": "Same evaluation text.", "metadata": {"tag": "dev"}}])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [{"id": "h", "text": "Same evaluation text.", "metadata": {"tag": "test"}}])
    experiment = replace(experiment, datasets=(
        DatasetSpec(str(screen), split="validation", purpose="screening"),
        DatasetSpec(str(holdout), split="test", purpose="confirmation"),
    ))
    with pytest.raises(ValueError, match="overlap"):
        run_experiment(experiment)


def test_oversized_qa_is_rejected_without_silent_truncation(tmp_path):
    experiment = _experiment(tmp_path)
    qa = _write_jsonl(tmp_path / "qa.jsonl", [{"question": "What is one plus one?", "choices": ["One", "Two"], "answer": 1}])
    experiment = replace(experiment, datasets=(DatasetSpec(str(qa), kind="qa", split="synthetic", purpose="smoke"),))
    with pytest.raises(ValueError, match="never silently truncated"):
        run_experiment(experiment)
    assert RunStore(experiment.output_dir).list()[0]["status"] == "failed"


@pytest.mark.parametrize("contradiction", [{"split": "test"}, {"purpose": "confirmation"}])
def test_source_metadata_cannot_silently_contradict_config(tmp_path, contradiction):
    experiment = _experiment(tmp_path)
    path = _write_jsonl(tmp_path / "tagged.jsonl", [
        {"_meta": {"split": "synthetic", "purpose": "smoke", **contradiction}},
        {"text": "A small fixture has a declared purpose."},
    ])
    experiment = replace(experiment, datasets=(DatasetSpec(str(path), split="synthetic", purpose="smoke"),))
    with pytest.raises(ValueError, match="disagrees"):
        run_experiment(experiment)
    assert RunStore(experiment.output_dir).list()[0]["status"] == "failed"


def test_source_metadata_matching_config_is_retained(tmp_path):
    experiment = _experiment(tmp_path)
    metadata = {"split": "synthetic", "purpose": "smoke", "source": "locally authored engineering fixture"}
    path = _write_jsonl(tmp_path / "tagged.jsonl", [
        {"_meta": metadata}, {"text": "A small fixture has a declared purpose."},
    ])
    experiment = replace(experiment, datasets=(DatasetSpec(str(path), split="synthetic", purpose="smoke"),))
    result = run_experiment(experiment)
    assert result["datasets"][0]["dataset"]["metadata"] == metadata


def test_config_resolves_dataset_and_output_relative_to_file(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"name": "relative", "model": {},
        "datasets": [{"path": "data.jsonl"}],
        "variants": [{"name": "candidate", "loop": {"start": 1, "end": 2}}],
        "output_dir": "outputs"}), encoding="utf-8")
    experiment = load_experiment(path)
    assert experiment.datasets[0].path == str(tmp_path / "data.jsonl")
    assert experiment.output_dir == str(tmp_path / "outputs")


@pytest.mark.parametrize("patch,error", [
    ({"unexpected": True}, "Unknown experiment fields"),
    ({"model": {"load_in_4bit": "false"}}, "booleans"),
    ({"datasets": [{"path": "x", "split": "test", "purpose": "screening"}]}, "reserve test"),
    ({"variants": [{"name": "candidate", "loop": {"start": 1, "end": 2, "steps": True}}]}, "integer"),
])
def test_config_rejects_mistyped_or_unknown_controls(patch, error):
    raw = {"name": "strict", "model": {}, "datasets": [{"path": "x"}],
           "variants": [{"name": "candidate", "loop": {"start": 1, "end": 2}}]}
    raw.update(patch)
    with pytest.raises(ValueError, match=error):
        experiment_from_dict(raw)
