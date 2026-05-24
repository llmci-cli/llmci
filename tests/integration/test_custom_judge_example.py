"""Integration test for the custom judge example."""

import os
from pathlib import Path

import pytest

from llmci.config import load_config
from llmci.runner import run_all_evals

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "examples" / "04-custom-judge"


@pytest.mark.asyncio
async def test_custom_judge_example():
    """Run the full pipeline on the 04-custom-judge example."""
    if not EXAMPLE_DIR.exists():
        pytest.skip("Example directory not found")

    orig_dir = os.getcwd()
    try:
        os.chdir(EXAMPLE_DIR)
        config = load_config()
        results = await run_all_evals(config)

        assert len(results) == 1
        result = results[0]
        assert result.eval_name == "api-json-validation"
        assert result.num_examples == 5
        assert result.num_errors == 0
        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] == 1.0
    finally:
        os.chdir(orig_dir)
