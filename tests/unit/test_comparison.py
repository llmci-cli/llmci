"""Tests for regression detection."""


from llmci.baseline import Baseline
from llmci.comparison import check_thresholds
from llmci.models import (
    EvalConfig,
    EvalResult,
    JudgeConfig,
    MetricThreshold,
)

_REG = {"name": "accuracy", "threshold": 0.05, "mode": "max_regression"}
_REG_TIGHT = {"name": "accuracy", "threshold": 0.02, "mode": "max_regression"}


def _result(name: str, metrics: dict[str, float]) -> EvalResult:
    return EvalResult(eval_name=name, metrics=metrics, num_examples=10)


def _config(name: str, thresholds: list[dict]) -> EvalConfig:
    return EvalConfig(
        name=name,
        dataset="test.jsonl",
        judge=JudgeConfig(type="exact_match"),
        metrics=[MetricThreshold(**t) for t in thresholds],
    )


def _baseline(name: str, metrics: dict[str, float]) -> Baseline:
    return Baseline(
        eval_name=name,
        metrics=metrics,
        timestamp="2025-01-01T00:00:00",
        commit_sha="abc123",
    )


class TestCheckThresholds:
    def test_absolute_pass(self):
        results = [_result("e", {"accuracy": 0.95})]
        configs = [_config("e", [{"name": "accuracy", "threshold": 0.9, "mode": "absolute"}])]
        trs = check_thresholds(results, {}, configs)
        assert len(trs) == 1
        assert trs[0].passed is True

    def test_absolute_fail(self):
        results = [_result("e", {"accuracy": 0.80})]
        configs = [_config("e", [{"name": "accuracy", "threshold": 0.9, "mode": "absolute"}])]
        trs = check_thresholds(results, {}, configs)
        assert trs[0].passed is False

    def test_max_regression_no_baseline(self):
        results = [_result("e", {"accuracy": 0.95})]
        configs = [_config("e", [_REG])]
        trs = check_thresholds(results, {}, configs)
        assert trs[0].passed is True
        assert "skipped" in trs[0].detail.lower()

    def test_max_regression_pass(self):
        results = [_result("e", {"accuracy": 0.94})]
        configs = [_config("e", [_REG])]
        baselines = {"e": _baseline("e", {"accuracy": 0.95})}
        trs = check_thresholds(results, baselines, configs)
        assert trs[0].passed is True

    def test_max_regression_fail(self):
        results = [_result("e", {"accuracy": 0.85})]
        configs = [_config("e", [_REG_TIGHT])]
        baselines = {"e": _baseline("e", {"accuracy": 0.95})}
        trs = check_thresholds(results, baselines, configs)
        assert trs[0].passed is False
        assert "Dropped" in trs[0].detail

    def test_max_regression_baseline_zero(self):
        results = [_result("e", {"accuracy": 0.0})]
        configs = [_config("e", [_REG])]
        baselines = {"e": _baseline("e", {"accuracy": 0.0})}
        trs = check_thresholds(results, baselines, configs)
        assert trs[0].passed is True

    def test_multiple_evals_and_metrics(self):
        results = [
            _result("e1", {"accuracy": 0.95, "f1": 0.90}),
            _result("e2", {"accuracy": 0.70}),
        ]
        configs = [
            _config("e1", [
                {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
                {"name": "f1", "threshold": 0.85, "mode": "absolute"},
            ]),
            _config("e2", [
                {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
            ]),
        ]
        trs = check_thresholds(results, {}, configs)
        assert len(trs) == 3
        assert trs[0].passed is True   # e1.accuracy
        assert trs[1].passed is True   # e1.f1
        assert trs[2].passed is False  # e2.accuracy

    def test_detail_string_format(self):
        results = [_result("e", {"accuracy": 0.85})]
        configs = [_config("e", [_REG_TIGHT])]
        baselines = {"e": _baseline("e", {"accuracy": 0.95})}
        trs = check_thresholds(results, baselines, configs)
        assert "0.950" in trs[0].detail
        assert "0.850" in trs[0].detail
        assert "%" in trs[0].detail
