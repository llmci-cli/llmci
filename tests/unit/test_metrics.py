"""Tests for aggregate metric computation."""

import pytest

from scaffold.metrics import _compute_classification_metrics, compute_metrics
from scaffold.models import EvalExample, JudgeResult, TargetResult


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
