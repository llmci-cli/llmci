"""Tests for pre-run gate configuration warnings."""

from llmci.gate_warnings import collect_gate_warnings
from llmci.models import (
    EvalConfig,
    JudgeConfig,
    LlmciConfig,
    MetricThreshold,
    Settings,
    TargetConfig,
)


def _config(**kwargs) -> LlmciConfig:
    defaults = {
        "target": TargetConfig(command="echo"),
        "evals": [EvalConfig(name="demo", dataset="data.jsonl")],
        "settings": Settings(),
    }
    defaults.update(kwargs)
    return LlmciConfig(**defaults)


class TestCollectGateWarnings:
    def test_no_warnings_for_minimal_config(self):
        assert collect_gate_warnings(_config()) == []

    def test_warns_sampling_without_significance(self):
        cfg = _config(settings=Settings(samples_per_example=3))
        warnings = collect_gate_warnings(cfg)
        assert len(warnings) == 1
        assert "significance" in warnings[0]

    def test_warns_max_regression_without_baseline(self):
        cfg = _config(evals=[
            EvalConfig(
                name="demo",
                dataset="data.jsonl",
                metrics=[MetricThreshold(name="accuracy", threshold=0.1, mode="max_regression")],
            )
        ])
        warnings = collect_gate_warnings(cfg, baselines={})
        assert any("max_regression" in w for w in warnings)

    def test_no_regression_warning_when_baseline_present(self):
        from llmci.baseline import Baseline

        cfg = _config(evals=[
            EvalConfig(
                name="demo",
                dataset="data.jsonl",
                metrics=[MetricThreshold(name="accuracy", threshold=0.1, mode="max_regression")],
            )
        ])
        baselines = {
            "demo": Baseline(
                eval_name="demo",
                metrics={"accuracy": 0.9},
                timestamp="2026-01-01T00:00:00Z",
                commit_sha="abc",
                examples=[],
            )
        }
        warnings = collect_gate_warnings(cfg, baselines)
        assert not any("max_regression" in w for w in warnings)

    def test_warns_pairwise_without_baseline(self):
        cfg = _config(evals=[
            EvalConfig(
                name="prefs",
                dataset="data.jsonl",
                judge=JudgeConfig(type="pairwise"),
            )
        ])
        warnings = collect_gate_warnings(cfg, baselines={})
        assert any("pairwise" in w for w in warnings)

    def test_skips_warnings_when_updating_baseline(self):
        cfg = _config(settings=Settings(samples_per_example=5))
        assert collect_gate_warnings(cfg, update_baseline=True) == []
