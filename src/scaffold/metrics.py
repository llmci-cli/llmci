"""Aggregate metric computation from per-example judge results."""

from __future__ import annotations

import math
import statistics
from collections import Counter

from scaffold.models import EvalExample, JudgeResult, TargetResult


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

    if "cosine_similarity" in requested:
        available["cosine_similarity"] = _mean_cosine_similarity(
            valid_expected, valid_actual
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
