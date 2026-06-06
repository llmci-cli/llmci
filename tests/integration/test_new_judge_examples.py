"""Integration tests for the safety, RAG, and plugin judge examples.

These exercise the newer judge types end-to-end through the real config loader and
runner, using deterministic command targets (no API keys required).
"""

from pathlib import Path

import pytest

from llmci.config import load_config
from llmci.runner import run_all_evals

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


@pytest.mark.asyncio
async def test_safety_pii_example(monkeypatch):
    """11-safety-pii: deterministic pii_leakage gate over a clean assistant."""
    monkeypatch.chdir(EXAMPLES_DIR / "11-safety-pii")
    config = load_config()
    results = await run_all_evals(config)

    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "safety-pii"
    assert result.num_examples == 5
    assert result.num_errors == 0
    # Every response is PII-free, so the deterministic criterion is perfect.
    assert result.metrics["pii_leakage"] == 1.0


@pytest.mark.asyncio
async def test_rag_retrieval_example(monkeypatch):
    """12-rag-retrieval: deterministic retrieval recall/precision @k."""
    monkeypatch.chdir(EXAMPLES_DIR / "12-rag-retrieval")
    config = load_config()
    results = await run_all_evals(config)

    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "rag-retrieval"
    assert result.num_examples == 4
    assert result.num_errors == 0
    assert result.metrics["retrieval_recall"] == 1.0
    assert result.metrics["retrieval_precision"] == pytest.approx(0.625)


@pytest.mark.asyncio
async def test_plugin_judge_example(monkeypatch):
    """13-plugin-judge: a config-declared local plugin registers a new judge type."""
    monkeypatch.chdir(EXAMPLES_DIR / "13-plugin-judge")
    config = load_config()

    # The plugins: list in llmci.yaml should have registered the judge on load.
    from llmci.plugins import registered_judge_types

    assert "json_schema" in registered_judge_types()

    results = await run_all_evals(config)
    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "json-schema"
    assert result.num_examples == 4
    assert result.num_errors == 0
    assert result.metrics["accuracy"] == 1.0
