"""Aggregate metric computation from per-example judge results."""

from __future__ import annotations

from collections import Counter

from scaffold.models import EvalExample, JudgeResult, TargetResult


def compute_metrics(
    examples: list[EvalExample],
    results: list[TargetResult],
    per_example: list[JudgeResult],
    requested: list[str],
) -> dict[str, float]:
    """Compute requested aggregate metrics from per-example results.

    Always available:
      - pass_rate: fraction with score >= 0.5
      - mean_score: average score
      - accuracy: fraction with score == 1.0

    Available when expected/actual are categorical labels:
      - f1_macro, f1_micro, f1_weighted
      - precision_macro, recall_macro
    """
    valid_indices = [i for i, r in enumerate(results) if r.error is None]
    if not valid_indices:
        return {name: 0.0 for name in requested}

    valid_scores = [per_example[i].score for i in valid_indices]
    valid_expected = [examples[i].expected.strip() for i in valid_indices]
    valid_actual = [results[i].output.strip() for i in valid_indices]

    available: dict[str, float] = {}

    available["pass_rate"] = sum(1 for s in valid_scores if s >= 0.5) / len(valid_scores)
    available["mean_score"] = sum(valid_scores) / len(valid_scores)
    available["accuracy"] = sum(1 for s in valid_scores if s == 1.0) / len(valid_scores)

    if _needs_classification_metrics(requested):
        f1_metrics = _compute_classification_metrics(valid_expected, valid_actual)
        available.update(f1_metrics)

    # Rubric pass rate (for LLM judge — same as pass_rate but explicit name)
    available["rubric_pass_rate"] = available["pass_rate"]

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
        "precision_macro", "recall_macro",
    }
    return bool(set(requested) & classification_metrics)


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

    # Macro averages
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

    # Micro averages
    total_tp = sum(d["tp"] for d in per_label.values())
    total_fp = sum(d["fp"] for d in per_label.values())
    total_fn = sum(d["fn"] for d in per_label.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Weighted averages (weighted by support = number of true instances per label)
    label_counts = Counter(expected)
    total = len(expected) if expected else 1
    weighted_f1 = sum(
        f1s[i] * label_counts.get(label, 0) / total for i, label in enumerate(labels)
    )

    return {
        "f1_macro": sum(f1s) / n_labels,
        "f1_micro": micro_f1,
        "f1_weighted": weighted_f1,
        "precision_macro": sum(precisions) / n_labels,
        "recall_macro": sum(recalls) / n_labels,
    }


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
