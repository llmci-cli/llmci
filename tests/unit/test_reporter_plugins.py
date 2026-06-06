"""Tests for the report-sink plugin API."""

import pytest

from llmci.errors import ConfigError
from llmci.models import EvalConfig, EvalResult, JudgeConfig
from llmci.plugins import (
    ReportContext,
    get_reporter,
    register_reporter,
    registered_reporter_names,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def _ctx(passed=True):
    return ReportContext(
        results=[EvalResult(eval_name="e", metrics={"accuracy": 1.0})],
        configs=[EvalConfig(name="e", dataset="d.jsonl", judge=JudgeConfig())],
        passed=passed,
        report_markdown="## report",
    )


class TestRegister:
    def test_register_and_get(self):
        def sink(ctx):
            pass

        register_reporter("webhook", sink)
        assert get_reporter("webhook") is sink
        assert "webhook" in registered_reporter_names()

    def test_empty_name_rejected(self):
        with pytest.raises(ConfigError):
            register_reporter("", lambda ctx: None)

    def test_non_callable_rejected(self):
        with pytest.raises(ConfigError):
            register_reporter("bad", 123)  # type: ignore[arg-type]

    def test_conflicting_reregistration_rejected(self):
        register_reporter("r", lambda ctx: None)
        with pytest.raises(ConfigError, match="already registered"):
            register_reporter("r", lambda ctx: None)

    def test_idempotent_same_fn(self):
        def sink(ctx):
            pass

        register_reporter("r", sink)
        register_reporter("r", sink)
        assert registered_reporter_names() == ["r"]

    def test_get_unknown_is_none(self):
        assert get_reporter("nope") is None


class TestContext:
    def test_sink_receives_results_and_passed(self):
        seen = {}

        def sink(ctx: ReportContext):
            seen["passed"] = ctx.passed
            seen["n"] = len(ctx.results)
            seen["md"] = ctx.report_markdown
            seen["eval"] = ctx.results[0].eval_name

        register_reporter("capture", sink)
        get_reporter("capture")(_ctx(passed=False))
        assert seen == {"passed": False, "n": 1, "md": "## report", "eval": "e"}
