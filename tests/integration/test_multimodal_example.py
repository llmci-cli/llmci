"""Integration test for the multimodal vision example (mocked LLM)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from llmci.config import load_config
from llmci.runner import run_all_evals

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]


@pytest.mark.asyncio
async def test_multimodal_vision_example(monkeypatch):
    """18-multimodal-vision: image field becomes litellm multimodal content."""
    monkeypatch.chdir(EXAMPLES_DIR / "18-multimodal-vision")
    config = load_config()
    captured: dict = {}

    async def _mock(**kwargs):
        captured.update(kwargs)
        return _Resp("red")

    with patch("llmci.targets.direct.litellm.acompletion", side_effect=_mock):
        results = await run_all_evals(config)

    assert len(results) == 1
    result = results[0]
    assert result.eval_name == "vision-color"
    assert result.metrics["accuracy"] == 1.0

    content = captured["messages"][0]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)
