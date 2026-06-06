"""Tests for the custom metric plugin API."""

import pytest

from llmci.comparison import check_thresholds
from llmci.errors import ConfigError
from llmci.metrics import BUILTIN_METRIC_NAMES, compute_metrics, is_lower_is_better
from llmci.models import (
    EvalConfig,
    EvalExample,
    EvalResult,
    JudgeConfig,
    JudgeResult,
    MetricThreshold,
    TargetResult,
)
from llmci.plugins import (
    MetricContext,
    get_metric_fn,
    metric_is_lower_is_better,
    register_metric,
    registered_metric_names,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def _exact_len(ctx: MetricContext) -> float:
    """Toy metric: fraction of valid outputs that are non-empty."""
    if not ctx.valid_indices:
        return 0.0
    non_empty = sum(1 for i in ctx.valid_indices if ctx.results[i].output.strip())
    return non_empty / len(ctx.valid_indices)


def _make_inputs(outputs, scores):
    examples = [EvalExample(input=f"q{i}", expected="") for i in range(len(outputs))]
    results = [TargetResult(output=o, latency_ms=1.0) for o in outputs]
    per_example = [JudgeResult(score=s) for s in scores]
    return examples, results, per_example


class TestRegister:
    def test_register_and_compute(self):
        register_metric("non_empty_rate", _exact_len)
        examples, results, per_example = _make_inputs(["a", "", "c"], [1.0, 0.0, 1.0])
        metrics = compute_metrics(examples, results, per_example, ["non_empty_rate"])
        assert metrics["non_empty_rate"] == pytest.approx(2 / 3)

    def test_builtin_collision_rejected(self):
        for builtin in list(BUILTIN_METRIC_NAMES)[:5]:
            with pytest.raises(ConfigError, match="built-in"):
                register_metric(builtin, _exact_len)

    def test_empty_name_rejected(self):
        with pytest.raises(ConfigError):
            register_metric("", _exact_len)

    def test_non_callable_rejected(self):
        with pytest.raises(ConfigError):
            register_metric("bad", 123)  # type: ignore[arg-type]

    def test_conflicting_reregistration_rejected(self):
        register_metric("m", _exact_len)
        with pytest.raises(ConfigError, match="already registered"):
            register_metric("m", lambda ctx: 0.0)

    def test_idempotent_same_fn(self):
        register_metric("m", _exact_len)
        register_metric("m", _exact_len)
        assert "m" in registered_metric_names()

    def test_get_metric_fn_unknown_is_none(self):
        assert get_metric_fn("nope") is None


class TestLowerIsBetter:
    def test_default_is_higher_better(self):
        register_metric("score_like", _exact_len)
        assert metric_is_lower_is_better("score_like") is False
        assert is_lower_is_better("score_like") is False

    def test_flag_marks_lower_is_better(self):
        register_metric("drift", lambda ctx: 0.0, lower_is_better=True)
        assert metric_is_lower_is_better("drift") is True
        assert is_lower_is_better("drift") is True


class TestComputeFallback:
    def test_unregistered_metric_falls_back_to_pass_rate(self):
        examples, results, per_example = _make_inputs(["a", "b"], [1.0, 0.0])
        metrics = compute_metrics(examples, results, per_example, ["mystery"])
        # pass_rate over [1.0, 0.0] is 0.5
        assert metrics["mystery"] == 0.5

    def test_broken_metric_does_not_crash(self):
        def explode(ctx):
            raise RuntimeError("boom")

        register_metric("explodes", explode)
        examples, results, per_example = _make_inputs(["a", "b"], [1.0, 1.0])
        metrics = compute_metrics(examples, results, per_example, ["explodes"])
        # Falls back to pass_rate (1.0) rather than raising.
        assert metrics["explodes"] == 1.0


class TestGatingWithDirection:
    def _check(self, threshold: float):
        register_metric("staleness", lambda ctx: 0.2, lower_is_better=True)
        result = EvalResult(eval_name="e", metrics={"staleness": 0.2})
        config = EvalConfig(
            name="e",
            dataset="d.jsonl",
            judge=JudgeConfig(type="exact_match"),
            metrics=[MetricThreshold(name="staleness", threshold=threshold, mode="absolute")],
        )
        return check_thresholds([result], {}, [config])

    def test_lower_is_better_passes_when_below_threshold(self):
        # 0.2 <= 0.3 passes for a lower-is-better metric.
        results = self._check(0.3)
        assert all(tr.passed for tr in results)

    def test_lower_is_better_fails_when_above_threshold(self):
        # 0.2 <= 0.1 fails.
        results = self._check(0.1)
        assert any(not tr.passed for tr in results)
