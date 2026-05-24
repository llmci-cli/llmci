"""Tests for report generation."""

from llmci.baseline import Baseline
from llmci.models import EvalConfig, EvalResult, JudgeConfig, MetricThreshold
from llmci.report import format_report


def _make_result(eval_name: str, metrics: dict[str, float], num_examples: int = 10) -> EvalResult:
    return EvalResult(
        eval_name=eval_name,
        metrics=metrics,
        num_examples=num_examples,
    )


def _make_config(eval_name: str, thresholds: list[dict]) -> EvalConfig:
    return EvalConfig(
        name=eval_name,
        dataset="test.jsonl",
        judge=JudgeConfig(type="exact_match"),
        metrics=[MetricThreshold(**t) for t in thresholds],
    )


def _make_baseline(eval_name: str, metrics: dict[str, float]) -> Baseline:
    return Baseline(
        eval_name=eval_name,
        metrics=metrics,
        timestamp="2025-01-01T00:00:00",
        commit_sha="abc123",
    )


class TestFormatReport:
    def test_all_pass(self):
        results = [_make_result("test", {"accuracy": 0.95})]
        configs = [_make_config("test", [
            {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
        ])]
        report, passed = format_report(results, configs)
        assert passed is True
        assert "✅" in report
        assert "❌" not in report

    def test_threshold_fail(self):
        results = [_make_result("test", {"accuracy": 0.80})]
        configs = [_make_config("test", [
            {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
        ])]
        report, passed = format_report(results, configs)
        assert passed is False
        assert "❌" in report

    def test_max_regression_skipped_without_baseline(self):
        results = [_make_result("test", {"accuracy": 0.95})]
        configs = [_make_config("test", [
            {"name": "accuracy", "threshold": 0.05, "mode": "max_regression"},
        ])]
        report, passed = format_report(results, configs)
        assert passed is True
        assert "⚠️" in report

    def test_max_regression_with_baseline_pass(self):
        results = [_make_result("test", {"accuracy": 0.94})]
        configs = [_make_config("test", [
            {"name": "accuracy", "threshold": 0.05, "mode": "max_regression"},
        ])]
        baselines = {"test": _make_baseline("test", {"accuracy": 0.95})}
        report, passed = format_report(results, configs, baselines=baselines)
        assert passed is True

    def test_max_regression_with_baseline_fail(self):
        results = [_make_result("test", {"accuracy": 0.85})]
        configs = [_make_config("test", [
            {"name": "accuracy", "threshold": 0.02, "mode": "max_regression"},
        ])]
        baselines = {"test": _make_baseline("test", {"accuracy": 0.95})}
        report, passed = format_report(results, configs, baselines=baselines)
        assert passed is False
        assert "❌" in report

    def test_report_contains_table(self):
        results = [_make_result("test", {"accuracy": 0.95})]
        configs = [_make_config("test", [
            {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
        ])]
        report, _ = format_report(results, configs)
        assert "| Eval" in report
        assert "| test" in report

    def test_baseline_report_has_baseline_column(self):
        results = [_make_result("test", {"accuracy": 0.94})]
        configs = [_make_config("test", [
            {"name": "accuracy", "threshold": 0.05, "mode": "max_regression"},
        ])]
        baselines = {"test": _make_baseline("test", {"accuracy": 0.95})}
        report, _ = format_report(results, configs, baselines=baselines)
        assert "Baseline" in report
        assert "This PR" in report
        assert "0.950" in report
        assert "0.940" in report

    def test_no_baseline_report_omits_baseline_column(self):
        results = [_make_result("test", {"accuracy": 0.95})]
        configs = [_make_config("test", [
            {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
        ])]
        report, _ = format_report(results, configs)
        assert "Baseline" not in report
