"""Tests for baseline-vs-current per-example output diffs."""

from llmci.baseline import Baseline, BaselineExample, save_baseline
from llmci.comparison import compute_output_diffs
from llmci.models import (
    EvalConfig,
    EvalExample,
    EvalResult,
    JudgeConfig,
    JudgeResult,
    MetricThreshold,
    TargetResult,
)
from llmci.report import format_report
from llmci.report_formats import format_report_as


def _result_with_examples(pairs) -> EvalResult:
    """pairs: list of (input, output, score)."""
    return EvalResult(
        eval_name="svc",
        metrics={"accuracy": sum(s for _, _, s in pairs) / len(pairs)},
        examples=[EvalExample(input=i, expected="") for i, _, _ in pairs],
        results=[TargetResult(output=o, latency_ms=1.0) for _, o, _ in pairs],
        per_example=[JudgeResult(score=s) for _, _, s in pairs],
        num_examples=len(pairs),
    )


def _baseline(pairs) -> Baseline:
    return Baseline(
        eval_name="svc",
        metrics={"accuracy": sum(s for _, _, s in pairs) / len(pairs)},
        timestamp="2025-01-01T00:00:00",
        commit_sha="abc",
        examples=[BaselineExample(input=i, output=o, score=s) for i, o, s in pairs],
    )


class TestComputeOutputDiffs:
    def test_only_regressions_returned(self):
        baseline = _baseline([("a", "good", 1.0), ("b", "good", 1.0)])
        result = _result_with_examples([("a", "bad", 0.0), ("b", "good", 1.0)])

        diffs = compute_output_diffs(result, baseline)
        assert len(diffs) == 1
        assert diffs[0].input == "a"
        assert diffs[0].baseline_output == "good"
        assert diffs[0].current_output == "bad"
        assert diffs[0].baseline_score == 1.0
        assert diffs[0].current_score == 0.0

    def test_matches_by_input_not_index(self):
        baseline = _baseline([("a", "A", 1.0), ("b", "B", 1.0)])
        # Order swapped in the current run.
        result = _result_with_examples([("b", "B", 1.0), ("a", "bad", 0.0)])

        diffs = compute_output_diffs(result, baseline)
        assert [d.input for d in diffs] == ["a"]

    def test_no_baseline_examples_returns_empty(self):
        baseline = Baseline(
            eval_name="svc", metrics={"accuracy": 1.0},
            timestamp="t", commit_sha="x", examples=[],
        )
        result = _result_with_examples([("a", "bad", 0.0)])
        assert compute_output_diffs(result, baseline) == []

    def test_sorted_worst_first(self):
        baseline = _baseline([("a", "A", 1.0), ("b", "B", 1.0)])
        result = _result_with_examples([("a", "x", 0.5), ("b", "y", 0.0)])
        diffs = compute_output_diffs(result, baseline)
        assert [d.input for d in diffs] == ["b", "a"]  # b dropped more


def _config() -> EvalConfig:
    return EvalConfig(
        name="svc", dataset="d.jsonl", judge=JudgeConfig(type="exact_match"),
        metrics=[MetricThreshold(name="accuracy", threshold=0.9, mode="absolute")],
    )


def test_markdown_report_shows_diffs():
    baseline = _baseline([("a", "good", 1.0)])
    result = _result_with_examples([("a", "bad", 0.0)])
    report, _ = format_report([result], [_config()], baselines={"svc": baseline})
    assert "Output Diffs vs Baseline" in report
    assert "good" in report and "bad" in report


def test_html_report_shows_diffs():
    baseline = _baseline([("a", "good", 1.0)])
    result = _result_with_examples([("a", "bad", 0.0)])
    content, _ = format_report_as("html", [result], [_config()], baselines={"svc": baseline})
    assert "Output Diffs vs Baseline" in content


def test_baseline_roundtrip_persists_examples(tmp_path, monkeypatch):
    monkeypatch.setattr("llmci.baseline.BASELINE_DIR", tmp_path)
    from llmci.baseline import load_baseline

    result = _result_with_examples([("a", "good", 1.0), ("b", "ok", 1.0)])
    save_baseline(result, commit_sha="abc")
    loaded = load_baseline("svc")

    assert loaded is not None
    assert len(loaded.examples) == 2
    assert loaded.examples[0].input == "a"
    assert loaded.examples[0].output == "good"


def test_load_tolerates_baseline_without_examples(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("llmci.baseline.BASELINE_DIR", tmp_path)
    from llmci.baseline import load_baseline

    (tmp_path / "svc.json").write_text(json.dumps({
        "eval_name": "svc", "metrics": {"accuracy": 0.9},
        "timestamp": "t", "commit_sha": "x",
    }))
    loaded = load_baseline("svc")
    assert loaded is not None
    assert loaded.examples == []
