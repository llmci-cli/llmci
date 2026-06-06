"""Tests for JUnit / SARIF / JSON report formats."""

import json
from xml.etree import ElementTree as ET

from llmci.baseline import Baseline
from llmci.models import EvalConfig, EvalResult, JudgeConfig, MetricThreshold
from llmci.report_formats import format_report_as


def _make_result(eval_name: str, metrics: dict, num_examples: int = 10) -> EvalResult:
    return EvalResult(eval_name=eval_name, metrics=metrics, num_examples=num_examples)


def _make_config(eval_name: str, thresholds: list[dict]) -> EvalConfig:
    return EvalConfig(
        name=eval_name,
        dataset="test.jsonl",
        judge=JudgeConfig(type="exact_match"),
        metrics=[MetricThreshold(**t) for t in thresholds],
    )


class TestJunit:
    def test_pass_produces_no_failures(self):
        results = [_make_result("clf", {"accuracy": 0.95})]
        configs = [_make_config("clf", [
            {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
        ])]
        content, passed = format_report_as("junit", results, configs)

        assert passed is True
        root = ET.fromstring(content)
        assert root.tag == "testsuites"
        assert root.get("failures") == "0"
        assert root.find("testsuite").get("name") == "clf"
        assert root.find(".//testcase").get("name") == "accuracy"
        assert root.find(".//failure") is None

    def test_fail_produces_failure_element(self):
        results = [_make_result("clf", {"accuracy": 0.50})]
        configs = [_make_config("clf", [
            {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
        ])]
        content, passed = format_report_as("junit", results, configs)

        assert passed is False
        root = ET.fromstring(content)
        assert root.get("failures") == "1"
        failure = root.find(".//failure")
        assert failure is not None
        assert "accuracy" in failure.get("message")

    def test_missing_baseline_regression_is_skipped(self):
        results = [_make_result("clf", {"accuracy": 0.95})]
        configs = [_make_config("clf", [
            {"name": "accuracy", "threshold": 0.05, "mode": "max_regression"},
        ])]
        content, passed = format_report_as("junit", results, configs)

        root = ET.fromstring(content)
        assert root.find(".//skipped") is not None
        assert passed is True


class TestSarif:
    def test_clean_run_has_empty_results(self):
        results = [_make_result("clf", {"accuracy": 0.95})]
        configs = [_make_config("clf", [
            {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
        ])]
        content, passed = format_report_as("sarif", results, configs)

        doc = json.loads(content)
        assert doc["version"] == "2.1.0"
        assert doc["runs"][0]["tool"]["driver"]["name"] == "llmci"
        assert doc["runs"][0]["results"] == []
        assert passed is True

    def test_failure_becomes_sarif_result(self):
        results = [_make_result("clf", {"accuracy": 0.50})]
        configs = [_make_config("clf", [
            {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
        ])]
        content, passed = format_report_as("sarif", results, configs)

        doc = json.loads(content)
        sarif_results = doc["runs"][0]["results"]
        assert len(sarif_results) == 1
        assert sarif_results[0]["ruleId"] == "llmci/accuracy"
        assert sarif_results[0]["level"] == "error"
        rules = doc["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["id"] == "llmci/accuracy"


class TestJson:
    def test_structure_with_baseline(self):
        results = [_make_result("clf", {"accuracy": 0.80})]
        configs = [_make_config("clf", [
            {"name": "accuracy", "threshold": 0.05, "mode": "max_regression"},
        ])]
        baselines = {
            "clf": Baseline(
                eval_name="clf",
                metrics={"accuracy": 0.90},
                timestamp="2025-01-01T00:00:00",
                commit_sha="abc",
            )
        }
        content, passed = format_report_as("json", results, configs, baselines=baselines)

        doc = json.loads(content)
        assert passed is False  # 0.90 -> 0.80 is ~11% drop > 5%
        eval_doc = doc["evals"][0]
        assert eval_doc["name"] == "clf"
        threshold = eval_doc["thresholds"][0]
        assert threshold["metric"] == "accuracy"
        assert threshold["baseline"] == 0.90
        assert threshold["passed"] is False


def test_markdown_delegates_to_report():
    results = [_make_result("clf", {"accuracy": 0.95})]
    configs = [_make_config("clf", [
        {"name": "accuracy", "threshold": 0.9, "mode": "absolute"},
    ])]
    content, passed = format_report_as("markdown", results, configs)
    assert "llmci Eval Report" in content
    assert passed is True
