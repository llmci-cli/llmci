"""Tests for the significance helpers and multi-sample aggregation."""

from llmci.comparison import check_thresholds
from llmci.models import (
    EvalConfig,
    EvalExample,
    EvalResult,
    JudgeConfig,
    JudgeResult,
    MetricThreshold,
    TargetResult,
)
from llmci.runner import _aggregate_rounds
from llmci.significance import confidence_interval, mean


class TestConfidenceInterval:
    def test_empty(self):
        assert confidence_interval([]) == (0.0, 0.0)

    def test_single_value_collapses(self):
        assert confidence_interval([0.9]) == (0.9, 0.9)

    def test_identical_values_have_zero_width(self):
        low, high = confidence_interval([0.8, 0.8, 0.8])
        assert high - low == 0.0
        assert low == high

    def test_spread_brackets_the_mean(self):
        low, high = confidence_interval([0.6, 0.8, 1.0], confidence=0.95)
        assert low < 0.8 < high

    def test_higher_confidence_is_wider(self):
        lo90, hi90 = confidence_interval([0.6, 0.8, 1.0], confidence=0.90)
        lo99, hi99 = confidence_interval([0.6, 0.8, 1.0], confidence=0.99)
        assert (hi99 - lo99) > (hi90 - lo90)


def test_mean():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == 2.0


def _round(scores: list[float]):
    results = [TargetResult(output="x", latency_ms=1.0) for _ in scores]
    per_example = [JudgeResult(score=s) for s in scores]
    return results, per_example


class TestAggregateRounds:
    def test_averages_metric_across_rounds(self):
        examples = [EvalExample(input="a", expected="x"), EvalExample(input="b", expected="y")]
        rounds = [_round([1.0, 1.0]), _round([1.0, 0.0])]

        metrics, metric_ci = _aggregate_rounds(
            examples, rounds, ["accuracy"], samples=2, significance=0.95
        )

        assert metrics["accuracy"] == 0.75  # mean of 1.0 and 0.5
        assert "accuracy" in metric_ci
        low, high = metric_ci["accuracy"]
        assert low < 0.75 < high

    def test_single_sample_has_no_ci(self):
        examples = [EvalExample(input="a", expected="x")]
        rounds = [_round([1.0])]

        metrics, metric_ci = _aggregate_rounds(
            examples, rounds, ["accuracy"], samples=1, significance=None
        )

        assert metrics["accuracy"] == 1.0
        assert metric_ci == {}


def _result(metrics, metric_ci=None, significance=None) -> EvalResult:
    return EvalResult(
        eval_name="clf",
        metrics=metrics,
        metric_ci=metric_ci or {},
        significance=significance,
        num_examples=10,
    )


def _config(threshold, mode="max_regression") -> EvalConfig:
    return EvalConfig(
        name="clf",
        dataset="test.jsonl",
        judge=JudgeConfig(type="exact_match"),
        metrics=[MetricThreshold(name="accuracy", threshold=threshold, mode=mode)],
    )


def _baseline(value):
    from llmci.baseline import Baseline

    return {"clf": Baseline(
        eval_name="clf", metrics={"accuracy": value},
        timestamp="2025-01-01T00:00:00", commit_sha="abc",
    )}


class TestSignificanceGating:
    def test_regression_within_noise_passes(self):
        # Baseline 0.90, current mean 0.88 (≈2% drop > 5%? no, < 5%). Use a wider gap
        # but a CI whose optimistic end keeps the drop under threshold.
        result = _result(
            {"accuracy": 0.85},
            metric_ci={"accuracy": (0.80, 0.92)},
            significance=0.95,
        )
        configs = [_config(threshold=0.05)]
        trs = check_thresholds([result], _baseline(0.90), configs)

        tr = trs[0]
        # Point drop is ~5.6% (> 5%), but optimistic end (0.92) implies a drop < 5%.
        assert tr.significant is False
        assert tr.passed is True
        assert "not significant" in tr.detail

    def test_regression_beyond_noise_fails(self):
        result = _result(
            {"accuracy": 0.70},
            metric_ci={"accuracy": (0.66, 0.74)},
            significance=0.95,
        )
        configs = [_config(threshold=0.05)]
        trs = check_thresholds([result], _baseline(0.90), configs)

        tr = trs[0]
        assert tr.significant is True
        assert tr.passed is False
        assert "significant at 95%" in tr.detail

    def test_no_significance_config_uses_point_estimate(self):
        result = _result({"accuracy": 0.80})  # no ci, no significance
        configs = [_config(threshold=0.05)]
        trs = check_thresholds([result], _baseline(0.90), configs)

        tr = trs[0]
        # ~11% drop > 5% with no significance gating → fail on the point estimate.
        assert tr.significant is None
        assert tr.passed is False
