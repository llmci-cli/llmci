"""Tests for cost/token metrics and lower-is-better threshold direction."""

from llmci.baseline import Baseline
from llmci.comparison import check_thresholds
from llmci.metrics import LOWER_IS_BETTER, compute_metrics
from llmci.models import (
    EvalConfig,
    EvalExample,
    EvalResult,
    JudgeConfig,
    JudgeResult,
    MetricThreshold,
    TargetResult,
)


def _examples(n: int) -> list[EvalExample]:
    return [EvalExample(input=f"i{i}", expected="x") for i in range(n)]


class TestCostMetrics:
    def test_cost_and_token_means(self):
        results = [
            TargetResult(output="a", latency_ms=1.0, tokens_in=100, tokens_out=20, cost=0.01),
            TargetResult(output="b", latency_ms=1.0, tokens_in=200, tokens_out=40, cost=0.03),
        ]
        per_example = [JudgeResult(score=1.0), JudgeResult(score=1.0)]
        metrics = compute_metrics(
            _examples(2), results, per_example,
            ["cost_total", "cost_mean", "tokens_in_mean", "tokens_out_mean", "tokens_total_mean"],
        )
        assert metrics["cost_total"] == 0.04
        assert metrics["cost_mean"] == 0.02
        assert metrics["tokens_in_mean"] == 150.0
        assert metrics["tokens_out_mean"] == 30.0
        assert metrics["tokens_total_mean"] == 180.0

    def test_errors_excluded(self):
        results = [
            TargetResult(output="a", latency_ms=1.0, tokens_in=100, tokens_out=20, cost=0.01),
            TargetResult(output="", latency_ms=0.0, error="boom", tokens_in=0, tokens_out=0),
        ]
        per_example = [JudgeResult(score=1.0), JudgeResult(score=0.0)]
        metrics = compute_metrics(
            _examples(2), results, per_example, ["cost_mean", "tokens_in_mean"],
        )
        assert metrics["cost_mean"] == 0.01
        assert metrics["tokens_in_mean"] == 100.0

    def test_no_valid_results_is_zero(self):
        results = [TargetResult(output="", latency_ms=0.0, error="boom")]
        per_example = [JudgeResult(score=0.0)]
        metrics = compute_metrics(_examples(1), results, per_example, ["cost_total"])
        assert metrics["cost_total"] == 0.0


def _result(metrics, eval_name="svc") -> EvalResult:
    return EvalResult(eval_name=eval_name, metrics=metrics, num_examples=10)


def _config(name, threshold, mode) -> EvalConfig:
    return EvalConfig(
        name="svc",
        dataset="d.jsonl",
        judge=JudgeConfig(type="exact_match"),
        metrics=[MetricThreshold(name=name, threshold=threshold, mode=mode)],
    )


class TestLowerIsBetterDirection:
    def test_cost_metrics_are_lower_is_better(self):
        assert "cost_mean" in LOWER_IS_BETTER
        assert "tokens_total_mean" in LOWER_IS_BETTER
        assert "latency_p90" in LOWER_IS_BETTER
        assert "accuracy" not in LOWER_IS_BETTER

    def test_absolute_passes_when_under_budget(self):
        trs = check_thresholds(
            [_result({"cost_mean": 0.01})], {}, [_config("cost_mean", 0.02, "absolute")]
        )
        assert trs[0].passed is True
        assert "≤" in trs[0].detail

    def test_absolute_fails_when_over_budget(self):
        trs = check_thresholds(
            [_result({"cost_mean": 0.05})], {}, [_config("cost_mean", 0.02, "absolute")]
        )
        assert trs[0].passed is False
        assert ">" in trs[0].detail

    def test_max_regression_flags_cost_increase(self):
        baseline = {"svc": Baseline(
            eval_name="svc", metrics={"cost_mean": 0.010},
            timestamp="2025-01-01T00:00:00", commit_sha="abc",
        )}
        # 20% increase, threshold 10% → regression.
        trs = check_thresholds(
            [_result({"cost_mean": 0.012})], baseline,
            [_config("cost_mean", 0.10, "max_regression")],
        )
        assert trs[0].passed is False
        assert "Rose" in trs[0].detail

    def test_max_regression_passes_when_cost_drops(self):
        baseline = {"svc": Baseline(
            eval_name="svc", metrics={"cost_mean": 0.020},
            timestamp="2025-01-01T00:00:00", commit_sha="abc",
        )}
        # Cost went down — never a regression for a lower-is-better metric.
        trs = check_thresholds(
            [_result({"cost_mean": 0.010})], baseline,
            [_config("cost_mean", 0.10, "max_regression")],
        )
        assert trs[0].passed is True
