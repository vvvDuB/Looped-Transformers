"""Numerical evaluation primitives for aligned, frozen-model experiments.

All probabilities use natural logarithms. Proximity is a descriptive distance,
not a quality score. Functions accept NumPy arrays and never mutate their inputs.
Every returned dictionary is serializable with ``json.dumps(..., allow_nan=False)``.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any

import numpy as np


_MAX_EXP = math.log(np.finfo(np.float64).max)


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _mask(mask: Any, count: int, name: str = "mask") -> np.ndarray:
    if mask is None:
        return np.ones(count, dtype=bool)
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.shape != (count,):
        raise ValueError(f"{name} must be a boolean vector of length {count}")
    if not array.any():
        raise ValueError(f"{name} must select at least one position")
    return array


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def log_softmax(logits: Any) -> np.ndarray:
    """Stable float64 log probabilities on the final (vocabulary) dimension."""
    values = _finite_array(logits, "logits")
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("logits must have a vocabulary dimension of at least 2")
    with np.errstate(over="raise", invalid="raise"):
        try:
            shifted = values - values.max(axis=-1, keepdims=True)
            result = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        except FloatingPointError as exc:
            raise ValueError("logit range exceeds float64 numerical limits") from exc
    return result


def _lm_result(nll_sum: float, count: int) -> dict[str, Any]:
    if not math.isfinite(nll_sum) or nll_sum < 0:
        raise ValueError("NLL exceeds float64 numerical limits or is negative")
    mean_nll = float(nll_sum / count)
    overflow = mean_nll > _MAX_EXP
    return {
        "nll_sum": float(nll_sum),
        "token_count": int(count),
        "mean_nll": mean_nll,
        "perplexity": None if overflow else float(math.exp(mean_nll)),
        "perplexity_overflow": bool(overflow),
    }


def language_model_metrics(
    logits: Any, token_ids: Sequence[int], mask: Any = None, *, chunk_size: int = 64
) -> dict[str, Any]:
    """Teacher-forced next-token NLL and perplexity, with an explicit one-token shift.

    ``logits`` has shape ``[sequence, vocabulary]``; logit row ``i`` predicts
    ``token_ids[i+1]``. ``mask`` is a boolean vector of length ``sequence - 1``
    selecting targets. For overlapping evaluation windows, use it to score each
    target once while retaining preceding tokens as context. Counts are tokens,
    not documents; use :func:`aggregate_language_model_metrics` across windows.
    """
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("logits must have shape [sequence >= 2, vocabulary >= 2]")
    ids = np.asarray(token_ids)
    if ids.ndim != 1 or ids.shape[0] != values.shape[0]:
        raise ValueError("token_ids must be a vector matching the logit sequence length")
    if ids.dtype.kind not in "iu" or ids.dtype == np.bool_:
        raise ValueError("token_ids must be integers")
    if np.any(ids < 0) or np.any(ids >= values.shape[1]):
        raise ValueError("token_ids contain an out-of-vocabulary index")
    selected = np.flatnonzero(_mask(mask, len(ids) - 1))
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    sums = []
    for start in range(0, len(selected), chunk_size):
        positions = selected[start : start + chunk_size]
        logp = log_softmax(values[positions])
        sums.append(float(-logp[np.arange(len(positions)), ids[positions + 1]].sum()))
    return _lm_result(math.fsum(sums), len(selected))


def aggregate_language_model_metrics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute corpus perplexity from summed NLL/count, never a mean of perplexities."""
    rows = list(items)
    if not rows:
        raise ValueError("at least one language-model result is required")
    counts, sums = [], []
    for row in rows:
        counts.append(_positive_integer(row["token_count"], "token_count"))
        nll = float(row["nll_sum"])
        if not math.isfinite(nll) or nll < 0:
            raise ValueError("nll_sum must be finite and nonnegative")
        sums.append(nll)
    return _lm_result(math.fsum(sums), sum(counts))


def logits_proximity(
    baseline: Any, candidate: Any, mask: Any = None, *, chunk_size: int = 64
) -> dict[str, Any]:
    """Full-vocabulary KL(P || Q), JS(P,Q), and top-1 agreement on aligned positions.

    Inputs have matching ``[..., vocabulary]`` shapes. The optional boolean mask
    indexes flattened non-vocabulary positions. No next-token shift is applied:
    callers select the positions whose output distributions they wish to compare.
    Vocabulary entries and token positions must have identical meanings.
    Computation is chunked over positions without a top-k approximation.
    """
    base, cand = np.asarray(baseline), np.asarray(candidate)
    if base.shape != cand.shape or base.ndim < 2 or base.shape[-1] < 2:
        raise ValueError("logits must have identical [..., vocabulary >= 2] shapes")
    base, cand = base.reshape(-1, base.shape[-1]), cand.reshape(-1, cand.shape[-1])
    if len(base) == 0:
        raise ValueError("logits must contain at least one position")
    positions = np.flatnonzero(_mask(mask, len(base)))
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    kl_parts, js_parts, agrees = [], [], 0
    for start in range(0, len(positions), chunk_size):
        idx = positions[start : start + chunk_size]
        p_log, q_log = log_softmax(base[idx]), log_softmax(cand[idx])
        p, q = np.exp(p_log), np.exp(q_log)
        m_log = np.logaddexp(p_log, q_log) - math.log(2.0)
        kl_parts.append(float(np.sum(p * (p_log - q_log))))
        js_parts.append(float(0.5 * np.sum(p * (p_log - m_log) + q * (q_log - m_log))))
        agrees += int(np.count_nonzero(np.argmax(base[idx], axis=-1) == np.argmax(cand[idx], axis=-1)))
    count = len(positions)
    # Roundoff can produce a tiny negative divergence when P and Q coincide.
    kl, js = max(0.0, math.fsum(kl_parts) / count), max(0.0, math.fsum(js_parts) / count)
    if not math.isfinite(kl) or not math.isfinite(js):
        raise ValueError("probability distance exceeds float64 numerical limits")
    return {
        "token_count": count,
        "kl_baseline_to_candidate": kl,
        "js_divergence": js,
        "top1_agreement": float(agrees / count),
    }


def aggregate_logits_proximity(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Token-weighted average of distances, including unequal window/document lengths."""
    rows = list(items)
    if not rows:
        raise ValueError("at least one proximity result is required")
    counts = [_positive_integer(row["token_count"], "token_count") for row in rows]
    total = sum(counts)
    result: dict[str, Any] = {"token_count": total}
    for key in ("kl_baseline_to_candidate", "js_divergence", "top1_agreement"):
        values = [float(row[key]) for row in rows]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"{key} must be finite and nonnegative")
        if key == "top1_agreement" and any(v > 1 for v in values):
            raise ValueError("top1_agreement must lie in [0, 1]")
        result[key] = math.fsum(v * n for v, n in zip(values, counts)) / total
    return result


def hidden_proximity(baseline: Any, candidate: Any, mask: Any = None) -> dict[str, Any]:
    """Compare matching hidden vectors with cosine and a global relative L2 norm.

    Both-zero vectors have cosine 1; exactly one zero vector has cosine 0.
    Relative L2 is ``||candidate-baseline||_F / ||baseline||_F`` across selected
    vectors. If the whole baseline is zero and the candidate is nonzero it is
    undefined, represented as JSON ``null`` with ``relative_l2_defined=False``.
    """
    base, cand = _finite_array(baseline, "baseline"), _finite_array(candidate, "candidate")
    if base.shape != cand.shape or base.ndim < 2:
        raise ValueError("hidden states must have identical [..., hidden_size] shapes")
    base, cand = base.reshape(-1, base.shape[-1]), cand.reshape(-1, cand.shape[-1])
    selected = _mask(mask, len(base))
    base, cand = base[selected], cand[selected]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            bnorm = np.linalg.norm(base, axis=-1)
            cnorm = np.linalg.norm(cand, axis=-1)
            cos = np.zeros(len(base), dtype=np.float64)
            both_nonzero = (bnorm > 0) & (cnorm > 0)
            cos[both_nonzero] = np.sum(
                (base[both_nonzero] / bnorm[both_nonzero, None])
                * (cand[both_nonzero] / cnorm[both_nonzero, None]), axis=-1
            )
            cos[(bnorm == 0) & (cnorm == 0)] = 1.0
            baseline_squared_norm = float(np.sum(base * base))
            difference_squared_norm = float(np.sum((cand - base) ** 2))
        except FloatingPointError as exc:
            raise ValueError("hidden-state norm exceeds float64 numerical limits") from exc
    defined = baseline_squared_norm > 0 or difference_squared_norm == 0
    relative = (
        float(math.sqrt(difference_squared_norm / baseline_squared_norm))
        if baseline_squared_norm > 0 else (0.0 if difference_squared_norm == 0 else None)
    )
    if relative is not None and not math.isfinite(relative):
        raise ValueError("relative hidden-state distance exceeds float64 numerical limits")
    return {
        "token_count": int(len(base)),
        "cosine_similarity": float(np.clip(cos, -1, 1).mean()),
        "relative_l2": relative,
        "relative_l2_defined": bool(defined),
        "baseline_squared_norm": baseline_squared_norm,
        "difference_squared_norm": difference_squared_norm,
        "zero_baseline_vectors": int(np.count_nonzero(bnorm == 0)),
        "zero_candidate_vectors": int(np.count_nonzero(cnorm == 0)),
    }


def aggregate_hidden_proximity(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate cosine by token count and relative L2 by additive squared norms."""
    rows = list(items)
    if not rows:
        raise ValueError("at least one hidden-state proximity result is required")
    counts = [_positive_integer(row["token_count"], "token_count") for row in rows]
    total = sum(counts)
    b2 = math.fsum(float(row["baseline_squared_norm"]) for row in rows)
    d2 = math.fsum(float(row["difference_squared_norm"]) for row in rows)
    if not math.isfinite(b2) or not math.isfinite(d2) or b2 < 0 or d2 < 0:
        raise ValueError("squared norms must be finite and nonnegative")
    cosines = [float(row["cosine_similarity"]) for row in rows]
    if any(not math.isfinite(c) or not -1 <= c <= 1 for c in cosines):
        raise ValueError("cosine_similarity must be finite and lie in [-1, 1]")
    relative = math.sqrt(d2 / b2) if b2 > 0 else (0.0 if d2 == 0 else None)
    if relative is not None and not math.isfinite(relative):
        raise ValueError("relative hidden-state distance exceeds float64 numerical limits")
    return {
        "token_count": total,
        "cosine_similarity": math.fsum(c * n for c, n in zip(cosines, counts)) / total,
        "relative_l2": relative,
        "relative_l2_defined": b2 > 0 or d2 == 0,
        "baseline_squared_norm": b2,
        "difference_squared_norm": d2,
        "zero_baseline_vectors": sum(int(r["zero_baseline_vectors"]) for r in rows),
        "zero_candidate_vectors": sum(int(r["zero_candidate_vectors"]) for r in rows),
    }


def _routing_events(events: Sequence[dict[str, Any]], label: str) -> dict[tuple[int, int], np.ndarray]:
    result = {}
    for event in events:
        layer, visit = event.get("layer"), event.get("visit")
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in (layer, visit)):
            raise ValueError(f"{label}: layer and visit must be nonnegative integers")
        ids = np.asarray(event.get("expert_ids"))
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError(f"{label}: expert_ids must be a nonempty [tokens, top_k] integer array")
        if int(ids.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{label}: expert ids exceed signed 64-bit range")
        ids = ids.astype(np.int64, copy=False)
        if any(len(set(row.tolist())) != len(row) for row in ids):
            raise ValueError(f"{label}: a token cannot select the same expert twice")
        key = (int(layer), int(visit))
        if key in result:
            raise ValueError(f"{label}: duplicate routing event for layer/visit {key}")
        result[key] = ids
    return result


def _load_distribution(ids: np.ndarray, num_experts: int, known_count: bool) -> dict[str, Any]:
    counts = np.bincount(ids.reshape(-1), minlength=num_experts)
    probabilities = counts[counts > 0] / float(counts.sum())
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {
        "expert_selection_counts": counts.tolist(),
        "load_entropy_nats": entropy,
        "normalized_load_entropy": (entropy / math.log(num_experts) if num_experts > 1 else 0.0) if known_count else None,
        "active_experts": int(np.count_nonzero(counts)),
    }


def routing_proximity(
    baseline_events: Sequence[dict[str, Any]],
    candidate_events: Sequence[dict[str, Any]],
    num_experts: int | None = None,
) -> dict[str, Any]:
    """Compare each candidate physical-layer visit to that layer's first baseline visit.

    A comparison requires equal token counts; positions are aligned by row.
    Top-k sizes may differ, so Jaccard uses sets. Entropy measures assignment load,
    not router probability entropy. Repeating a layer does not freeze its routing.
    ``token_count`` here counts *layer-visit-token comparisons*, not corpus tokens.
    """
    base, cand = _routing_events(baseline_events, "baseline"), _routing_events(candidate_events, "candidate")
    if num_experts is not None:
        num_experts = _positive_integer(num_experts, "num_experts")
    if not base or not cand:
        return {
            "available": False,
            "reason": "missing_baseline_routing" if not base else "missing_candidate_routing",
            "token_count": 0,
            "mean_topk_jaccard": None,
            "per_layer_visit": [],
            "unmatched_candidate_layers": sorted({layer for layer, _ in cand} - {layer for layer, _ in base}),
            "unmatched_baseline_layers": sorted({layer for layer, _ in base} - {layer for layer, _ in cand}),
        }
    known_count = num_experts is not None
    observed_count = max(int(ids.max()) for ids in [*base.values(), *cand.values()]) + 1
    if num_experts is None:
        num_experts = observed_count
    elif observed_count > num_experts:
        raise ValueError("routing contains an expert id outside num_experts")
    canonical = {}
    for (layer, visit), ids in sorted(base.items()):
        canonical.setdefault(layer, (visit, ids))
    results, unmatched = [], []
    for (layer, visit), cids in sorted(cand.items()):
        if layer not in canonical:
            unmatched.append(layer)
            continue
        base_visit, bids = canonical[layer]
        if bids.shape[0] != cids.shape[0]:
            raise ValueError(f"routing token counts differ at layer {layer}, visit {visit}")
        intersections = np.array([len(set(b.tolist()) & set(c.tolist())) for b, c in zip(bids, cids)], dtype=np.float64)
        jaccard = intersections / (bids.shape[1] + cids.shape[1] - intersections)
        results.append({
            "layer": layer,
            "baseline_visit": base_visit,
            "candidate_visit": visit,
            "token_count": int(len(cids)),
            "baseline_top_k": int(bids.shape[1]),
            "candidate_top_k": int(cids.shape[1]),
            "mean_topk_jaccard": float(jaccard.mean()),
            "baseline_load": _load_distribution(bids, num_experts, known_count),
            "candidate_load": _load_distribution(cids, num_experts, known_count),
        })
    count = sum(r["token_count"] for r in results)
    missing_candidate = sorted({layer for layer, _ in base} - {layer for layer, _ in cand})
    return {
        "available": bool(results),
        "reason": None if results else "no_matching_layers",
        "complete": not unmatched and not missing_candidate,
        "comparison_unit": "physical_layer_visit_token",
        "token_count": count,
        "mean_topk_jaccard": math.fsum(r["mean_topk_jaccard"] * r["token_count"] for r in results) / count if count else None,
        "num_experts": int(num_experts),
        "num_experts_source": "provided" if known_count else "inferred_from_observed_ids",
        "per_layer_visit": results,
        "unmatched_candidate_layers": sorted(set(unmatched)),
        "unmatched_baseline_layers": missing_candidate,
    }


def paired_bootstrap(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Percentile bootstrap for paired candidate-minus-baseline mean differences.

    Resampling units are paired examples/documents. Optional positive weights
    (e.g. target counts with per-document mean NLL) are resampled alongside each
    pair, preserving token-weighted means. The interval describes sampling
    uncertainty, not repeated kernel timing or hyperparameter-search uncertainty.
    """
    base = _finite_array(baseline_values, "baseline_values", ndim=1)
    cand = _finite_array(candidate_values, "candidate_values", ndim=1)
    if base.shape != cand.shape:
        raise ValueError("paired arrays must have equal lengths")
    samples = _positive_integer(samples, "samples")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    w = np.ones(len(base), dtype=np.float64) if weights is None else _finite_array(weights, "weights", ndim=1)
    if w.shape != base.shape or np.any(w <= 0):
        raise ValueError("weights must be positive and match the paired arrays")
    # Scaling leaves the ratio unchanged and prevents large target counts from
    # overflowing when multiplied into scores.
    w = w / w.max()
    if np.any(w == 0):
        raise ValueError("weight range exceeds float64 numerical limits")
    delta = cand - base
    if not np.isfinite(delta).all():
        raise ValueError("paired difference exceeds float64 numerical limits")
    estimate = float(np.average(delta, weights=w))
    random = np.random.default_rng(int(seed))
    replicates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 64):
        size = min(64, samples - start)
        idx = random.integers(0, len(base), size=(size, len(base)))
        selected_weights = w[idx]
        replicates[start : start + size] = np.sum(delta[idx] * selected_weights, axis=1) / selected_weights.sum(axis=1)
    if not math.isfinite(estimate) or not np.isfinite(replicates).all():
        raise ValueError("bootstrap mean exceeds float64 numerical limits")
    baseline_mean, candidate_mean = float(np.average(base, weights=w)), float(np.average(cand, weights=w))
    if not math.isfinite(baseline_mean) or not math.isfinite(candidate_mean):
        raise ValueError("paired mean exceeds float64 numerical limits")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(replicates, [tail, 1.0 - tail])
    return {
        "paired_count": int(len(base)),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "delta": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "bootstrap_samples": samples,
        "seed": int(seed),
        "method": "paired_percentile_bootstrap",
        "weighted": weights is not None,
        "insufficient_pairs": len(base) < 2,
    }


def paired_accuracy(
    baseline_correct: Sequence[bool], candidate_correct: Sequence[bool], **bootstrap_kwargs: Any
) -> dict[str, Any]:
    """Accuracy change and discordant-pair counts with a paired bootstrap interval."""
    base, cand = np.asarray(baseline_correct), np.asarray(candidate_correct)
    if base.ndim != 1 or cand.shape != base.shape or base.size == 0:
        raise ValueError("correctness arrays must be nonempty aligned vectors")
    if not np.isin(base, [0, 1]).all() or not np.isin(cand, [0, 1]).all():
        raise ValueError("correctness arrays must contain only booleans or 0/1")
    base, cand = base.astype(bool), cand.astype(bool)
    if "weights" in bootstrap_kwargs:
        raise ValueError("paired_accuracy uses one vote per example; weights are not accepted")
    result = paired_bootstrap(base.astype(float), cand.astype(float), **bootstrap_kwargs)
    result.update({
        "baseline_accuracy": float(base.mean()),
        "candidate_accuracy": float(cand.mean()),
        "delta_percentage_points": 100.0 * result["delta"],
        "both_correct": int(np.count_nonzero(base & cand)),
        "both_incorrect": int(np.count_nonzero(~base & ~cand)),
        "improved": int(np.count_nonzero(~base & cand)),
        "degraded": int(np.count_nonzero(base & ~cand)),
    })
    return result
