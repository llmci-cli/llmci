"""Tests for aggregate metric computation."""

import pytest

from llmci.metrics import (
    _compute_classification_metrics,
    _mean_cosine_similarity,
    compute_metrics,
)
from llmci.models import EvalExample, JudgeResult, TargetResult


def _make_data(pairs: list[tuple[str, str, float]]):
    """Helper: list of (expected, actual, score) → examples, results, judge_results."""
    examples = [EvalExample(input=f"q{i}", expected=exp) for i, (exp, _, _) in enumerate(pairs)]
    results = [TargetResult(output=act, latency_ms=10.0) for _, act, _ in pairs]
    per_example = [JudgeResult(score=sc) for _, _, sc in pairs]
    return examples, results, per_example


class TestComputeMetrics:
    def test_accuracy_all_correct(self):
        examples, results, per_example = _make_data([
            ("a", "a", 1.0),
            ("b", "b", 1.0),
            ("c", "c", 1.0),
        ])
        metrics = compute_metrics(examples, results, per_example, ["accuracy"])
        assert metrics["accuracy"] == 1.0

    def test_accuracy_half_correct(self):
        examples, results, per_example = _make_data([
            ("a", "a", 1.0),
            ("b", "x", 0.0),
        ])
        metrics = compute_metrics(examples, results, per_example, ["accuracy"])
        assert metrics["accuracy"] == 0.5

    def test_pass_rate(self):
        examples, results, per_example = _make_data([
            ("a", "a", 0.8),
            ("b", "b", 0.3),
            ("c", "c", 0.6),
        ])
        metrics = compute_metrics(examples, results, per_example, ["pass_rate"])
        assert metrics["pass_rate"] == pytest.approx(2 / 3)

    def test_f1_macro_perfect(self):
        examples, results, per_example = _make_data([
            ("a", "a", 1.0),
            ("b", "b", 1.0),
            ("a", "a", 1.0),
            ("b", "b", 1.0),
        ])
        metrics = compute_metrics(examples, results, per_example, ["f1_macro"])
        assert metrics["f1_macro"] == 1.0

    def test_f1_macro_imperfect(self):
        examples, results, per_example = _make_data([
            ("a", "a", 1.0),
            ("a", "b", 0.0),
            ("b", "b", 1.0),
            ("b", "a", 0.0),
        ])
        metrics = compute_metrics(examples, results, per_example, ["f1_macro"])
        assert 0.0 < metrics["f1_macro"] < 1.0

    def test_errors_excluded(self):
        examples = [
            EvalExample(input="q1", expected="a"),
            EvalExample(input="q2", expected="b"),
        ]
        results = [
            TargetResult(output="a", latency_ms=10.0),
            TargetResult(output="", latency_ms=0.0, error="timeout"),
        ]
        per_example = [JudgeResult(score=1.0), JudgeResult(score=0.0)]
        metrics = compute_metrics(examples, results, per_example, ["accuracy"])
        assert metrics["accuracy"] == 1.0  # only 1 valid, and it's correct

    def test_all_errors(self):
        examples = [EvalExample(input="q", expected="a")]
        results = [TargetResult(output="", latency_ms=0.0, error="fail")]
        per_example = [JudgeResult(score=0.0)]
        metrics = compute_metrics(examples, results, per_example, ["accuracy"])
        assert metrics["accuracy"] == 0.0

    def test_median_score(self):
        examples, results, per_example = _make_data([
            ("a", "a", 0.2),
            ("b", "b", 0.5),
            ("c", "c", 0.9),
        ])
        metrics = compute_metrics(examples, results, per_example, ["median_score"])
        assert metrics["median_score"] == 0.5

    def test_min_max_score(self):
        examples, results, per_example = _make_data([
            ("a", "a", 0.1),
            ("b", "b", 0.7),
            ("c", "c", 0.3),
        ])
        metrics = compute_metrics(
            examples, results, per_example, ["min_score", "max_score"]
        )
        assert metrics["min_score"] == pytest.approx(0.1)
        assert metrics["max_score"] == pytest.approx(0.7)

    def test_error_rate(self):
        examples = [
            EvalExample(input="q1", expected="a"),
            EvalExample(input="q2", expected="b"),
            EvalExample(input="q3", expected="c"),
            EvalExample(input="q4", expected="d"),
        ]
        results = [
            TargetResult(output="a", latency_ms=10.0),
            TargetResult(output="", latency_ms=0.0, error="timeout"),
            TargetResult(output="c", latency_ms=10.0),
            TargetResult(output="", latency_ms=0.0, error="fail"),
        ]
        per_example = [
            JudgeResult(score=1.0), JudgeResult(score=0.0),
            JudgeResult(score=1.0), JudgeResult(score=0.0),
        ]
        metrics = compute_metrics(examples, results, per_example, ["error_rate"])
        assert metrics["error_rate"] == 0.5

    def test_error_rate_no_errors(self):
        examples, results, per_example = _make_data([
            ("a", "a", 1.0), ("b", "b", 1.0),
        ])
        metrics = compute_metrics(examples, results, per_example, ["error_rate"])
        assert metrics["error_rate"] == 0.0

    def test_latency_metrics(self):
        examples = [
            EvalExample(input=f"q{i}", expected="a") for i in range(10)
        ]
        results = [
            TargetResult(output="a", latency_ms=float(i * 100))
            for i in range(10)
        ]
        per_example = [JudgeResult(score=1.0) for _ in range(10)]
        metrics = compute_metrics(
            examples, results, per_example,
            ["latency_mean", "latency_p50", "latency_p90", "latency_p99"],
        )
        assert metrics["latency_mean"] == pytest.approx(450.0)
        assert metrics["latency_p50"] == pytest.approx(500.0)
        assert metrics["latency_p90"] > metrics["latency_p50"]

    def test_cosine_similarity_identical(self):
        examples, results, per_example = _make_data([
            ("hello world", "hello world", 1.0),
        ])
        metrics = compute_metrics(
            examples, results, per_example, ["cosine_similarity"]
        )
        assert metrics["cosine_similarity"] == pytest.approx(1.0)

    def test_cosine_similarity_disjoint(self):
        examples, results, per_example = _make_data([
            ("cat dog", "fish bird", 0.0),
        ])
        metrics = compute_metrics(
            examples, results, per_example, ["cosine_similarity"]
        )
        assert metrics["cosine_similarity"] == pytest.approx(0.0)

    def test_multiple_metrics_at_once(self):
        examples, results, per_example = _make_data([
            ("a", "a", 1.0),
            ("b", "b", 0.6),
            ("c", "x", 0.0),
        ])
        metrics = compute_metrics(
            examples, results, per_example,
            ["accuracy", "mean_score", "median_score", "pass_rate",
             "min_score", "max_score", "error_rate"],
        )
        assert metrics["accuracy"] == pytest.approx(1 / 3)
        assert metrics["mean_score"] == pytest.approx(1.6 / 3)
        assert metrics["median_score"] == pytest.approx(0.6)
        assert metrics["pass_rate"] == pytest.approx(2 / 3)
        assert metrics["min_score"] == pytest.approx(0.0)
        assert metrics["max_score"] == pytest.approx(1.0)
        assert metrics["error_rate"] == 0.0


class TestClassificationMetrics:
    def test_binary_perfect(self):
        metrics = _compute_classification_metrics(
            ["a", "b", "a", "b"], ["a", "b", "a", "b"]
        )
        assert metrics["f1_macro"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0

    def test_multiclass(self):
        metrics = _compute_classification_metrics(
            ["a", "b", "c", "a", "b", "c"],
            ["a", "b", "c", "a", "b", "c"],
        )
        assert metrics["f1_macro"] == 1.0

    def test_all_wrong(self):
        metrics = _compute_classification_metrics(
            ["a", "b"], ["b", "a"]
        )
        assert metrics["f1_macro"] == 0.0

    def test_micro_metrics_perfect(self):
        metrics = _compute_classification_metrics(
            ["a", "b", "a", "b"], ["a", "b", "a", "b"]
        )
        assert metrics["precision_micro"] == 1.0
        assert metrics["recall_micro"] == 1.0
        assert metrics["f1_micro"] == 1.0

    def test_weighted_metrics_perfect(self):
        metrics = _compute_classification_metrics(
            ["a", "b", "a", "b"], ["a", "b", "a", "b"]
        )
        assert metrics["precision_weighted"] == 1.0
        assert metrics["recall_weighted"] == 1.0
        assert metrics["f1_weighted"] == 1.0

    def test_micro_vs_macro_imbalanced(self):
        expected = ["a"] * 9 + ["b"]
        actual = ["a"] * 9 + ["a"]
        metrics = _compute_classification_metrics(expected, actual)
        assert metrics["precision_micro"] != metrics["precision_macro"]
        assert metrics["recall_micro"] != metrics["recall_macro"]

    def test_all_metrics_present(self):
        metrics = _compute_classification_metrics(
            ["a", "b"], ["a", "b"]
        )
        expected_keys = {
            "f1_macro", "f1_micro", "f1_weighted",
            "precision_macro", "precision_micro", "precision_weighted",
            "recall_macro", "recall_micro", "recall_weighted",
        }
        assert set(metrics.keys()) == expected_keys


class TestCosineSimilarity:
    def test_identical(self):
        assert _mean_cosine_similarity(["hello world"], ["hello world"]) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _mean_cosine_similarity(["cat dog"], ["fish bird"]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        sim = _mean_cosine_similarity(["the cat sat"], ["the dog sat"])
        assert 0.0 < sim < 1.0

    def test_case_insensitive(self):
        assert _mean_cosine_similarity(["Hello World"], ["hello world"]) == pytest.approx(1.0)

    def test_empty_strings(self):
        assert _mean_cosine_similarity([""], ["hello"]) == pytest.approx(0.0)
        assert _mean_cosine_similarity(["hello"], [""]) == pytest.approx(0.0)

    def test_multiple_examples(self):
        sim = _mean_cosine_similarity(
            ["hello world", "cat dog"],
            ["hello world", "fish bird"],
        )
        assert sim == pytest.approx(0.5)
