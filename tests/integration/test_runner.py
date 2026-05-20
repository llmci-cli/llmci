"""Integration tests for the full eval runner."""

from pathlib import Path

import pytest

from scaffold.config import load_config
from scaffold.runner import run_all_evals

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "examples" / "01-ci-regression"


@pytest.mark.asyncio
async def test_full_pipeline_on_example():
    """Run the full pipeline on the 01-ci-regression example."""
    if not EXAMPLE_DIR.exists():
        pytest.skip("Example directory not found")

    import os
    orig_dir = os.getcwd()
    try:
        os.chdir(EXAMPLE_DIR)
        config = load_config()
        results = await run_all_evals(config)

        assert len(results) == 1
        result = results[0]
        assert result.eval_name == "ticket-classification"
        assert result.num_examples == 20
        assert result.num_errors == 0
        assert "accuracy" in result.metrics
        assert "f1_macro" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert 0.0 <= result.metrics["f1_macro"] <= 1.0
    finally:
        os.chdir(orig_dir)


@pytest.mark.asyncio
async def test_smoke_mode_on_example():
    """Run with smoke test mode — fewer examples."""
    if not EXAMPLE_DIR.exists():
        pytest.skip("Example directory not found")

    import os
    orig_dir = os.getcwd()
    try:
        os.chdir(EXAMPLE_DIR)
        config = load_config()
        config.settings.smoke_test_size = 5
        results = await run_all_evals(config, smoke=True)

        assert len(results) == 1
        assert results[0].num_examples == 5
    finally:
        os.chdir(orig_dir)
