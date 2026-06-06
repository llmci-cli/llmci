"""Integration test for the stacked Now-tier CI gate example."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from llmci.baseline import load_all_baselines
from llmci.cli import cli
from llmci.comparison import check_thresholds
from llmci.config import load_config
from llmci.runner import run_all_evals

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "17-integrated-ci-gate"


@pytest.mark.asyncio
async def test_integrated_ci_gate_passes_with_baselines(monkeypatch):
    """17-integrated-ci-gate: quality + cost regression + safety in one config."""
    monkeypatch.chdir(EXAMPLE)
    config = load_config()
    baselines = load_all_baselines([e.name for e in config.evals])

    assert "support-routing" in baselines
    assert "support-safety" in baselines

    results = await run_all_evals(config, baselines=baselines)
    assert len(results) == 2

    routing = next(r for r in results if r.eval_name == "support-routing")
    safety = next(r for r in results if r.eval_name == "support-safety")

    assert routing.num_examples == 8
    assert routing.num_errors == 0
    assert routing.metrics["accuracy"] == 1.0
    assert routing.metrics["cost_mean"] > 0
    assert routing.metrics["tokens_in_mean"] > 0

    assert safety.metrics["pii_leakage"] == 1.0

    thresholds = check_thresholds(results, baselines, config.evals)
    assert all(tr.passed for tr in thresholds)


def test_integrated_example_cli_loads_local_baselines(monkeypatch):
    """CLI auto-loads committed .llmci/baselines/ when --compare-to is omitted."""
    monkeypatch.chdir(EXAMPLE)
    runner = CliRunner()
    result = runner.invoke(cli, ["-v", "run", "--config", "llmci.yaml"])
    assert result.exit_code == 0, result.output
    assert "Loaded 2 baseline(s) from .llmci/baselines/" in result.output
    assert "cost_mean" in result.output
    assert "pii_leakage" in result.output
