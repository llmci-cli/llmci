"""Aggregate metric computation from per-example judge results."""

from __future__ import annotations

import math
import statistics
from collections import Counter

from llmci.models import EvalExample, JudgeResult, TargetResult

# Metrics where a lower value is better. Threshold checks invert their comparison
# for these (absolute: current <= threshold; max_regression: an *increase* is the
# regression).
LOWER_IS_BETTER = frozenset({
    "error_rate",
    "latency_mean", "latency_p50", "latency_p90", "latency_p99",
    "cost_total", "cost_mean",
    "tokens_in_mean", "tokens_out_mean", "tokens_total_mean",
})

# All metric names llmci computes itself. Plugin metrics may not shadow these.
BUILTIN_METRIC_NAMES = frozenset({
    "pass_rate", "mean_score", "median_score", "min_score", "max_score",
    "accuracy", "error_rate", "rubric_pass_rate",
    "f1_macro", "f1_micro", "f1_weighted",
    "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro", "recall_micro", "recall_weighted",
    "latency_mean", "latency_p50", "latency_p90", "latency_p99",
    "cost_total", "cost_mean",
    "tokens_in_mean", "tokens_out_mean", "tokens_total_mean",
    "cosine_similarity",
})


def is_lower_is_better(name: str) -> bool:
    """Whether a metric (built-in or plugin-registered) is lower-is-better."""
    if name in LOWER_IS_BETTER:
        return True
    from llmci.plugins import metric_is_lower_is_better

    return metric_is_lower_is_better(name)


def compute_metrics(
    examples: list[EvalExample],
    results: list[TargetResult],
    per_example: list[JudgeResult],
    requested: list[str],
) -> dict[str, float]:
    """Compute requested aggregate metrics from per-example results.

    Always available (score-based):
      - pass_rate: fraction with score >= 0.5
      - mean_score: average score
      - median_score: median score
      - min_score: lowest score in dataset
      - max_score: highest score in dataset
      - accuracy: fraction with score == 1.0
      - error_rate: fraction of examples that errored

    Classification (expected/actual label pairs):
      - f1_macro, f1_micro, f1_weighted
      - precision_macro, precision_micro, precision_weighted
      - recall_macro, recall_micro, recall_weighted

    Latency:
      - latency_mean, latency_p50, latency_p90, latency_p99

    Cost / tokens (lower is better):
      - cost_total, cost_mean
      - tokens_in_mean, tokens_out_mean, tokens_total_mean

    Similarity:
      - cosine_similarity (token-overlap proxy)
    """
    n_total = len(results)
    valid_indices = [i for i, r in enumerate(results) if r.error is None]
    if not valid_indices:
        return {name: 0.0 for name in requested}

    valid_scores = [per_example[i].score for i in valid_indices]
    valid_expected = [examples[i].expected.strip() for i in valid_indices]
    valid_actual = [results[i].output.strip() for i in valid_indices]

    available: dict[str, float] = {}

    available["pass_rate"] = sum(1 for s in valid_scores if s >= 0.5) / len(valid_scores)
    available["mean_score"] = sum(valid_scores) / len(valid_scores)
    available["median_score"] = statistics.median(valid_scores)
    available["min_score"] = min(valid_scores)
    available["max_score"] = max(valid_scores)
    available["accuracy"] = sum(1 for s in valid_scores if s == 1.0) / len(valid_scores)

    n_errors = n_total - len(valid_indices)
    available["error_rate"] = n_errors / n_total if n_total > 0 else 0.0

    available["rubric_pass_rate"] = available["pass_rate"]

    if _needs_classification_metrics(requested):
        cls_metrics = _compute_classification_metrics(valid_expected, valid_actual)
        available.update(cls_metrics)

    if _needs_latency_metrics(requested):
        lat_metrics = _compute_latency_metrics(results)
        available.update(lat_metrics)

    if _needs_cost_metrics(requested):
        cost_metrics = _compute_cost_metrics(results)
        available.update(cost_metrics)

    if "cosine_similarity" in requested:
        available["cosine_similarity"] = _mean_cosine_similarity(
            valid_expected, valid_actual
        )

    # Per-example named sub-scores (e.g. RAG faithfulness) become aggregate metrics
    # by name. Built-ins above take precedence on any name collision.
    for name, value in _compute_subscore_metrics(per_example, valid_indices).items():
        available.setdefault(name, value)

    # Any still-unresolved requested metric may be provided by a plugin.
    unresolved = [name for name in requested if name not in available]
    if unresolved:
        available.update(
            _compute_plugin_metrics(
                unresolved, examples, results, per_example, valid_indices, valid_scores
            )
        )

    output = {}
    for name in requested:
        if name in available:
            output[name] = available[name]
        else:
            output[name] = available.get("pass_rate", 0.0)

    return output


def _needs_classification_metrics(requested: list[str]) -> bool:
    """Check if any requested metric requires classification computation."""
    classification_metrics = {
        "f1_macro", "f1_micro", "f1_weighted",
        "precision_macro", "precision_micro", "precision_weighted",
        "recall_macro", "recall_micro", "recall_weighted",
    }
    return bool(set(requested) & classification_metrics)


def _needs_latency_metrics(requested: list[str]) -> bool:
    latency_metrics = {"latency_mean", "latency_p50", "latency_p90", "latency_p99"}
    return bool(set(requested) & latency_metrics)


def _needs_cost_metrics(requested: list[str]) -> bool:
    cost_metrics = {
        "cost_total", "cost_mean",
        "tokens_in_mean", "tokens_out_mean", "tokens_total_mean",
    }
    return bool(set(requested) & cost_metrics)


def _compute_plugin_metrics(
    names: list[str],
    examples: list[EvalExample],
    results: list[TargetResult],
    per_example: list[JudgeResult],
    valid_indices: list[int],
    valid_scores: list[float],
) -> dict[str, float]:
    """Resolve requested metrics from the plugin registry, if registered."""
    from llmci.plugins import MetricContext, get_metric_fn

    ctx: MetricContext | None = None
    computed: dict[str, float] = {}
    for name in names:
        fn = get_metric_fn(name)
        if fn is None:
            continue
        if ctx is None:
            ctx = MetricContext(
                examples=examples,
                results=results,
                per_example=per_example,
                valid_indices=valid_indices,
                scores=valid_scores,
            )
        try:
            computed[name] = float(fn(ctx))
        except Exception:
            # A broken plugin metric shouldn't crash the run; fall through to the
            # pass_rate default applied by the caller.
            continue
    return computed


def _compute_subscore_metrics(
    per_example: list[JudgeResult],
    valid_indices: list[int],
) -> dict[str, float]:
    """Average each named per-example sub-score across valid examples."""
    names: set[str] = set()
    for i in valid_indices:
        names |= set(per_example[i].sub_scores.keys())

    metrics: dict[str, float] = {}
    for name in names:
        values = [
            per_example[i].sub_scores[name]
            for i in valid_indices
            if name in per_example[i].sub_scores
        ]
        if values:
            metrics[name] = sum(values) / len(values)
    return metrics


def _compute_cost_metrics(results: list[TargetResult]) -> dict[str, float]:
    """Compute cost and token-usage metrics from successful target results."""
    valid = [r for r in results if r.error is None]
    if not valid:
        return {
            "cost_total": 0.0, "cost_mean": 0.0,
            "tokens_in_mean": 0.0, "tokens_out_mean": 0.0, "tokens_total_mean": 0.0,
        }
    n = len(valid)
    cost_total = sum(r.cost for r in valid)
    tokens_in_total = sum(r.tokens_in for r in valid)
    tokens_out_total = sum(r.tokens_out for r in valid)
    return {
        "cost_total": cost_total,
        "cost_mean": cost_total / n,
        "tokens_in_mean": tokens_in_total / n,
        "tokens_out_mean": tokens_out_total / n,
        "tokens_total_mean": (tokens_in_total + tokens_out_total) / n,
    }


def _compute_classification_metrics(
    expected: list[str], actual: list[str]
) -> dict[str, float]:
    """Compute F1, precision, recall from expected/actual label pairs."""
    labels = sorted(set(expected) | set(actual))

    per_label: dict[str, dict[str, int]] = {}
    for label in labels:
        tp = sum(1 for e, a in zip(expected, actual) if e == label and a == label)
        fp = sum(1 for e, a in zip(expected, actual) if e != label and a == label)
        fn = sum(1 for e, a in zip(expected, actual) if e == label and a != label)
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn}

    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        tp = per_label[label]["tp"]
        fp = per_label[label]["fp"]
        fn = per_label[label]["fn"]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    n_labels = len(labels) if labels else 1

    total_tp = sum(d["tp"] for d in per_label.values())
    total_fp = sum(d["fp"] for d in per_label.values())
    total_fn = sum(d["fn"] for d in per_label.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )

    label_counts = Counter(expected)
    total = len(expected) if expected else 1
    weighted_f1 = sum(
        f1s[i] * label_counts.get(label, 0) / total
        for i, label in enumerate(labels)
    )
    weighted_p = sum(
        precisions[i] * label_counts.get(label, 0) / total
        for i, label in enumerate(labels)
    )
    weighted_r = sum(
        recalls[i] * label_counts.get(label, 0) / total
        for i, label in enumerate(labels)
    )

    return {
        "f1_macro": sum(f1s) / n_labels,
        "f1_micro": micro_f1,
        "f1_weighted": weighted_f1,
        "precision_macro": sum(precisions) / n_labels,
        "precision_micro": micro_p,
        "precision_weighted": weighted_p,
        "recall_macro": sum(recalls) / n_labels,
        "recall_micro": micro_r,
        "recall_weighted": weighted_r,
    }


def _compute_latency_metrics(results: list[TargetResult]) -> dict[str, float]:
    """Compute latency metrics usable as threshold-gated eval metrics."""
    latencies = sorted(r.latency_ms for r in results if r.error is None)
    if not latencies:
        return {
            "latency_mean": 0.0, "latency_p50": 0.0,
            "latency_p90": 0.0, "latency_p99": 0.0,
        }
    n = len(latencies)
    return {
        "latency_mean": sum(latencies) / n,
        "latency_p50": latencies[int(n * 0.5)],
        "latency_p90": latencies[min(int(n * 0.9), n - 1)],
        "latency_p99": latencies[min(int(n * 0.99), n - 1)],
    }


def _mean_cosine_similarity(expected: list[str], actual: list[str]) -> float:
    """Token-overlap cosine similarity averaged across examples.

    Uses bag-of-words (lowercased, whitespace-split) as a lightweight
    proxy — no embedding model needed.
    """
    similarities: list[float] = []
    for exp, act in zip(expected, actual):
        exp_tokens = exp.lower().split()
        act_tokens = act.lower().split()
        if not exp_tokens or not act_tokens:
            similarities.append(0.0)
            continue

        vocab = set(exp_tokens) | set(act_tokens)
        exp_counts = Counter(exp_tokens)
        act_counts = Counter(act_tokens)

        dot = sum(exp_counts[w] * act_counts[w] for w in vocab)
        mag_exp = math.sqrt(sum(c * c for c in exp_counts.values()))
        mag_act = math.sqrt(sum(c * c for c in act_counts.values()))

        if mag_exp == 0 or mag_act == 0:
            similarities.append(0.0)
        else:
            similarities.append(dot / (mag_exp * mag_act))

    return sum(similarities) / len(similarities) if similarities else 0.0


def compute_latency_stats(results: list[TargetResult]) -> dict[str, float]:
    """Compute latency percentiles from target results."""
    latencies = sorted(r.latency_ms for r in results if r.error is None)
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0}

    n = len(latencies)
    return {
        "p50": latencies[int(n * 0.5)],
        "p90": latencies[min(int(n * 0.9), n - 1)],
        "p99": latencies[min(int(n * 0.99), n - 1)],
        "mean": sum(latencies) / n,
    }
