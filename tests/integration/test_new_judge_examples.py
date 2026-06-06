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

    # The plugins: list in llmci.yaml should have registered the judge and metric on load.
    from llmci.plugins import registered_judge_types, registered_metric_names

    assert "json_schema" in registered_judge_types()
    assert "json_field_coverage" in registered_metric_names()

    results = await run_all_evals(config)
    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "json-schema"
    assert result.num_examples == 4
    assert result.num_errors == 0
    assert result.metrics["accuracy"] == 1.0
    # Custom metric plugin computed and surfaced by name.
    assert result.metrics["json_field_coverage"] == 1.0


@pytest.mark.asyncio
async def test_redteam_example(monkeypatch):
    """15-redteam: a generated adversarial dataset gated by the safety judge."""
    monkeypatch.chdir(EXAMPLES_DIR / "15-redteam")
    config = load_config()
    results = await run_all_evals(config)

    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "redteam-pii"
    # The committed attacks.jsonl is 2 seeds x 5 templates (pii_extraction + injection).
    assert result.num_examples == 10
    assert result.num_errors == 0
    # The assistant resists every adversarial framing, so no PII leaks.
    assert result.metrics["pii_leakage"] == 1.0


@pytest.mark.asyncio
async def test_structured_output_example(monkeypatch):
    """16-structured-output: validate JSON output against an inline JSON Schema."""
    monkeypatch.chdir(EXAMPLES_DIR / "16-structured-output")
    config = load_config()
    results = await run_all_evals(config)

    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "product-extraction"
    assert result.num_examples == 4
    assert result.num_errors == 0
    # Every extracted record conforms to the schema.
    assert result.metrics["accuracy"] == 1.0


def test_redteam_generation_is_deterministic(tmp_path, monkeypatch):
    """Regenerating from the example seeds reproduces the committed dataset exactly."""
    example = EXAMPLES_DIR / "15-redteam"
    from llmci.redteam import generate_attacks, load_seeds

    seeds = load_seeds(example / "seeds.txt")
    rows = generate_attacks(
        seeds, categories=["pii_extraction", "injection"]
    )
    committed = load_seeds(example / "evals" / "attacks.jsonl")
    assert [r["input"] for r in rows] == committed
