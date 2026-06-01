"""Integration tests for pipeline and service examples."""

from pathlib import Path

import pytest

from llmci.config import load_config
from llmci.runner import run_all_evals

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


@pytest.mark.asyncio
async def test_pipeline_level_example(monkeypatch):
    """Run the 07-pipeline-level example end to end."""
    monkeypatch.chdir(EXAMPLES_DIR / "07-pipeline-level")
    config = load_config()
    results = await run_all_evals(config)

    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "qa-pipeline"
    assert result.num_examples == 6
    assert result.num_errors == 0
    assert 0.0 <= result.metrics["pass_rate"] <= 1.0


@pytest.mark.asyncio
async def test_fastapi_service_level_example(monkeypatch):
    """Run the 08-fastapi-service full service example."""
    monkeypatch.chdir(EXAMPLES_DIR / "08-fastapi-service")
    config = load_config()
    results = await run_all_evals(config)

    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "service-classification"
    assert result.num_examples == 24
    assert result.num_errors == 0
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["f1_weighted"] <= 1.0


@pytest.mark.asyncio
async def test_fastapi_prompt_level_example(monkeypatch):
    """Run the 08-fastapi-service prompt-only config."""
    monkeypatch.chdir(EXAMPLES_DIR / "08-fastapi-service")
    config = load_config(Path("llmci-prompt-level.yaml"))
    results = await run_all_evals(config)

    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "prompt-classification"
    assert result.num_examples == 24
    assert result.num_errors == 0
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["f1_macro"] <= 1.0
