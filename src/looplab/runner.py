"""Paired, cache-free model experiments with provenance and honest cost accounting."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .datasets import load_dataset
from .experiment import Experiment
from .inference import ModelManager
from .metrics import (aggregate_hidden_proximity, aggregate_language_model_metrics, aggregate_logits_proximity,
                      hidden_proximity, language_model_metrics, log_softmax,
                      logits_proximity, paired_accuracy, paired_bootstrap,
                      routing_proximity)
from .store import RunStore, write_json


def text_windows(ids: list[int], context: int, stride: int):
    """Yield (tokens, prediction mask), scoring each noninitial target exactly once."""
    if not 0 < stride < context:
        raise ValueError("stride must be positive and smaller than context")
    for target_start in range(1, len(ids), stride):
        end = min(len(ids), target_start + stride)
        begin = max(0, end - context)
        tokens = ids[begin:end]
        mask = np.arange(begin + 1, end) >= target_start
        yield tokens, mask


def _environment() -> dict:
    packages = {}
    for package in ("looplab", "numpy", "torch", "transformers", "accelerate",
                    "bitsandbytes", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=3, cwd=Path(__file__).parent)
        revision = git.stdout.strip() if git.returncode == 0 else None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=3, cwd=Path(__file__).parent)
        dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        revision, dirty = None, None
    return {"python": sys.version, "platform": platform.platform(),
            "packages": packages, "git_revision": revision, "git_dirty": dirty}


def _forward(backend, ids, loop, trace):
    started = time.perf_counter()
    output = backend.forward(ids, loop=loop, trace=trace)
    elapsed = time.perf_counter() - started
    if not np.all(np.isfinite(output.logits)):
        raise ValueError("Model produced non-finite logits")
    return output, elapsed


def _new_accumulator():
    return {"logits": [], "hidden": {}, "routing": [], "forward_seconds": 0.0,
            "forward_count": 0, "input_token_evaluations": 0}


def _routing_mask(events, position_mask):
    # Only scored token positions contribute; overlapping context isn't counted again.
    result = []
    for event in events:
        value = dict(event)
        ids = np.asarray(event["expert_ids"])
        if len(ids) != len(position_mask):
            raise ValueError("Router trace is not aligned to input positions")
        value["expert_ids"] = ids[position_mask]
        result.append(value)
    return result


def _accumulate(acc, baseline, candidate, mask, elapsed, num_experts):
    acc["forward_seconds"] += elapsed
    acc["forward_count"] += 1
    acc["input_token_evaluations"] += len(candidate.logits)
    acc["logits"].append(logits_proximity(baseline.logits, candidate.logits, mask=mask))
    if set(baseline.hidden_states) != set(candidate.hidden_states):
        raise ValueError("Hidden-state trace layers differ between paired forwards")
    for name in baseline.hidden_states:
        metric = hidden_proximity(baseline.hidden_states[name], candidate.hidden_states[name], mask=mask)
        acc["hidden"].setdefault(name, []).append(metric)
    routing = routing_proximity(_routing_mask(baseline.routing, mask),
                                _routing_mask(candidate.routing, mask), num_experts=num_experts)
    acc["routing"].append(routing)


def _finish_accumulator(acc):
    return {"logits": aggregate_logits_proximity(acc["logits"]),
            "hidden_states": {k: aggregate_hidden_proximity(v) for k, v in acc["hidden"].items()},
            "routing_by_window": acc["routing"],
            "forward_seconds": acc["forward_seconds"], "forward_count": acc["forward_count"],
            "input_token_evaluations": acc["input_token_evaluations"]}


def _evaluate_text(backend, dataset, experiment, progress):
    variants = experiment.variants
    metrics = {"baseline": []} | {v.name: [] for v in variants}
    proximity = {v.name: _new_accumulator() for v in variants}
    per_document = []
    baseline_seconds = 0.0
    baseline_count = 0
    stride = experiment.stride or max(1, backend.max_context // 2)
    num_experts = backend.info().get("num_experts") or None
    for index, row in enumerate(dataset.rows):
        ids = backend.tokenize(row["text"])
        if len(ids) < 2:
            raise ValueError(f"Document {row['id']} has fewer than two tokens")
        document = {name: [] for name in metrics}
        for tokens, mask in text_windows(ids, backend.max_context, stride):
            base, elapsed = _forward(backend, tokens, None, experiment.trace)
            baseline_seconds += elapsed
            baseline_count += 1
            document["baseline"].append(language_model_metrics(base.logits, tokens, mask=mask))
            position_mask = np.append(mask, False)
            for variant in variants:
                output, elapsed = _forward(backend, tokens, variant.loop, experiment.trace)
                document[variant.name].append(language_model_metrics(output.logits, tokens, mask=mask))
                _accumulate(proximity[variant.name], base, output, position_mask, elapsed, num_experts)
        combined = {name: aggregate_language_model_metrics(value) for name, value in document.items()}
        for name in metrics:
            metrics[name].append(combined[name])
        per_document.append({"id": row["id"], "token_count": len(ids) - 1, "metrics": combined})
        progress(f"{dataset.path}: text {index + 1}/{len(dataset.rows)}")
    totals = {name: aggregate_language_model_metrics(items) for name, items in metrics.items()}
    base_values = [v["mean_nll"] for v in metrics["baseline"]]
    comparisons = {}
    for variant in variants:
        name = variant.name
        comparisons[name] = _finish_accumulator(proximity[name])
        comparisons[name]["mean_nll_delta"] = totals[name]["mean_nll"] - totals["baseline"]["mean_nll"]
        comparisons[name]["token_weighted_mean_nll_paired_ci"] = paired_bootstrap(
            base_values, [v["mean_nll"] for v in metrics[name]],
            samples=experiment.bootstrap_samples, seed=experiment.seed,
            weights=[v["token_count"] for v in metrics["baseline"]])
        comparisons[name]["time_ratio"] = (proximity[name]["forward_seconds"] / baseline_seconds
                                            if baseline_seconds else None)
    return {"kind": "text", "metrics": totals, "comparisons": comparisons,
            "baseline_forward_seconds": baseline_seconds, "baseline_forward_count": baseline_count,
            "stride": stride, "per_item": per_document,
            "ci_unit": "paired resampling of documents with token-count weighting; delta in corpus mean NLL, not PPL"}


def _qa_prompt(row):
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = "\n".join(f"{labels[i]}. {answer}" for i, answer in enumerate(row["choices"]))
    return f"Question: {row['question']}\n{choices}\nAnswer:"


def _evaluate_qa(backend, dataset, experiment, progress):
    names = ["baseline"] + [v.name for v in experiment.variants]
    correct = {name: [] for name in names}
    times = {name: 0.0 for name in names}
    counts = {name: 0 for name in names}
    per_item = []
    for index, row in enumerate(dataset.rows):
        prefix = backend.tokenize(_qa_prompt(row))
        if not prefix:
            raise ValueError("QA prompt tokenizes to empty input")
        scores = {name: [] for name in names}
        for choice_index in range(len(row["choices"])):
            suffix = backend.tokenize(" " + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[choice_index])
            if not suffix:
                raise ValueError("QA answer label tokenizes to empty input")
            tokens = prefix + suffix
            if len(tokens) > backend.max_context:
                raise ValueError(f"QA item {row['id']} needs {len(tokens)} tokens; max_context={backend.max_context}. "
                                 "Increase the context; questions are never silently truncated.")
            for name, loop in [("baseline", None)] + [(v.name, v.loop) for v in experiment.variants]:
                output, elapsed = _forward(backend, tokens, loop, False)
                times[name] += elapsed
                counts[name] += 1
                logits = output.logits[len(prefix) - 1:len(tokens) - 1]
                lp = log_softmax(logits)
                scores[name].append(float(lp[np.arange(len(suffix)), suffix].sum()))
        predictions = {name: int(np.argmax(values)) for name, values in scores.items()}
        for name in names:
            correct[name].append(int(predictions[name] == row["answer"]))
        per_item.append({"id": row["id"], "answer": row["answer"], "scores": scores, "predictions": predictions})
        progress(f"{dataset.path}: QA {index + 1}/{len(dataset.rows)}")
    metrics = {name: {"accuracy": sum(values) / len(values), "correct": sum(values),
                      "examples": len(values), "forward_seconds": times[name], "forward_count": counts[name]}
               for name, values in correct.items()}
    comparisons = {v.name: paired_accuracy(correct["baseline"], correct[v.name],
                     samples=experiment.bootstrap_samples, seed=experiment.seed)
                   for v in experiment.variants}
    return {"kind": "qa", "metrics": metrics, "comparisons": comparisons, "per_item": per_item,
            "prompt_policy": "plain completion; prompt and space-plus-answer-label tokenized separately; sum label log probabilities",
            "template_status": "custom harness, not an official model leaderboard protocol"}


def run_experiment(experiment: Experiment, manager: ModelManager | None = None,
                   progress=None) -> dict:
    progress = progress or (lambda message: None)
    store = RunStore(experiment.output_dir)
    run_id = store.start(experiment.name, experiment.as_dict())
    folder = store.path(run_id)
    started = time.perf_counter()
    own_manager = manager is None
    manager = manager or ModelManager()
    try:
        datasets = [load_dataset(s.path, kind=s.kind, limit=s.limit) for s in experiment.datasets]
        # Exact cross-file duplicates must not masquerade as held-out confirmation.
        screening = set()
        confirmation = set()
        for spec, dataset in zip(experiment.datasets, datasets):
            for key in ("split", "purpose"):
                if key in dataset.metadata and dataset.metadata[key] != getattr(spec, key):
                    raise ValueError(f"Dataset {spec.path} {key} disagrees with experiment configuration")
            fields = ("text",) if spec.kind == "text" else ("question", "choices", "answer")
            hashes = {hashlib.sha256(json.dumps({k: row[k] for k in fields},
                                               sort_keys=True).encode()).hexdigest() for row in dataset.rows}
            if spec.purpose == "screening":
                screening.update(hashes)
            elif spec.purpose == "confirmation":
                confirmation.update(hashes)
        if screening & confirmation:
            raise ValueError("Screening and confirmation datasets overlap exactly")
        with manager.lease(experiment.model) as backend:
            info = backend.info()
            num_layers = info.get("num_layers")
            for variant in experiment.variants:
                variant.loop.validate(num_layers)
            probe = backend.tokenize("A baseline probe.")[:min(16, backend.max_context)]
            baseline = backend.forward(probe, trace=False).logits.copy()
            # Exercise both loop modes' identity paths before accepting measurements.
            for variant in experiment.variants:
                identity = backend.forward(probe, loop=replace(variant.loop, steps=1), trace=False).logits
                if not np.array_equal(baseline, identity):
                    raise RuntimeError(f"K=1 identity failed for {variant.name}")
                backend.forward(probe, loop=variant.loop, trace=False)  # warm exact variant
                restored = backend.forward(probe, trace=False).logits
                if not np.array_equal(baseline, restored):
                    raise RuntimeError(f"Baseline restoration failed for {variant.name}")
            if hasattr(backend, "reset_peak_memory_stats"):
                backend.reset_peak_memory_stats()
            manifests = []
            for spec, dataset in zip(experiment.datasets, datasets):
                manifests.append(dataset.manifest() | {"split": spec.split, "purpose": spec.purpose,
                                                       "requested_limit": spec.limit})
            provenance = {"run_id": run_id, "environment": _environment(), "model": info,
                          "datasets": manifests, "experiment_sha256": hashlib.sha256(
                              json.dumps(experiment.as_dict(), sort_keys=True).encode()).hexdigest()}
            write_json(folder / "manifest.json", provenance)
            outcomes = []
            for index, (spec, dataset) in enumerate(zip(experiment.datasets, datasets)):
                progress(f"Evaluating {spec.kind}: {spec.path}")
                outcome = (_evaluate_text(backend, dataset, experiment, progress) if spec.kind == "text"
                           else _evaluate_qa(backend, dataset, experiment, progress))
                outcome["dataset"] = manifests[index]
                write_json(folder / f"dataset-{index}.json", outcome)
                compact = {k: v for k, v in outcome.items() if k != "per_item"}
                for comparison in compact.get("comparisons", {}).values():
                    # Keep detailed routing traces in per-dataset artifacts, not the run index.
                    if "routing_by_window" in comparison:
                        comparison["routing_windows"] = len(comparison.pop("routing_by_window"))
                outcomes.append(compact)
            restored = backend.forward(probe, trace=False).logits
            if not np.array_equal(baseline, restored):
                raise RuntimeError("Post-experiment baseline restoration failed")
            info = backend.info()
            provenance["measurement"] = {"model_after_evaluation": info,
                                         "memory_scope": "model residency plus evaluation after warmup reset",
                                         "trace_positions": "predictor positions for uniquely scored targets"}
            write_json(folder / "manifest.json", provenance)
            summary = {"schema_version": 1, "run_id": run_id, "name": experiment.name,
                       "status": "completed", "model": info, "datasets": outcomes,
                       "variants": [{"name": v.name, "loop": asdict(v.loop),
                                     "additional_layer_evaluations": (v.loop.end - v.loop.start + 1) * (v.loop.steps - 1),
                                     "layer_evaluation_ratio": 1 + (v.loop.end - v.loop.start + 1) * (v.loop.steps - 1) / num_layers}
                                    for v in experiment.variants],
                       "checks": {"k1_identity": True, "baseline_restored": True,
                                  "weight_updates": False, "cache_enabled": False},
                       "wall_seconds": time.perf_counter() - started,
                       "timing_scope": "warmed forward including tracing and host copies; diagnostic, not production decode",
                       "statistics": "paired document/item bootstrap, no multiplicity correction; configuration selection requires independent confirmation",
                       "artifact_directory": str(folder)}
        store.finish(run_id, summary)
        return summary
    except Exception as exc:
        store.fail(run_id, f"{type(exc).__name__}: {exc}",
                   status="cancelled" if isinstance(exc, InterruptedError) else "failed")
        raise
    finally:
        if own_manager:
            manager.unload()
