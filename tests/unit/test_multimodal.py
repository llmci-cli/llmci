"""Tests for multimodal direct-target message building."""

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from llmci.cache import ResponseCache
from llmci.models import EvalExample
from llmci.multimodal import (
    build_user_content,
    dataset_media_base,
    has_media,
    media_cache_params,
)
from llmci.targets.direct import run_direct_target

# 1x1 PNG (red pixel)
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class TestMultimodalHelpers:
    def test_dataset_media_base_from_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base = dataset_media_base("evals/data.jsonl")
        assert base == tmp_path / "evals"

    def test_has_media_detects_images(self):
        ex = EvalExample(input="q", expected="", extra={"images": ["a.png"]})
        assert has_media(ex)

    def test_build_text_only_returns_string(self):
        ex = EvalExample(input="hello", expected="")
        assert build_user_content("Prompt: hello", ex, Path("/tmp")) == "Prompt: hello"

    def test_build_image_content(self, tmp_path):
        img = tmp_path / "dot.png"
        img.write_bytes(_TINY_PNG)
        ex = EvalExample(input="describe", expected="", extra={"images": ["dot.png"]})
        content = build_user_content("Look:", ex, tmp_path)
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Look:"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_build_audio_content(self, tmp_path):
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"RIFFxxxxWAVEfmt ")
        ex = EvalExample(input="transcribe", expected="", extra={"audio": ["clip.wav"]})
        content = build_user_content("Listen:", ex, tmp_path)
        assert content[-1]["type"] == "input_audio"
        assert content[-1]["input_audio"]["format"] == "wav"
        assert content[-1]["input_audio"]["data"]

    def test_missing_file_raises(self, tmp_path):
        ex = EvalExample(input="q", expected="", extra={"images": ["nope.png"]})
        with pytest.raises(ValueError, match="not found"):
            build_user_content("x", ex, tmp_path)


class _MockMessage:
    def __init__(self, content):
        self.content = content


class _MockResponse:
    def __init__(self):
        self.choices = [type("C", (), {"message": _MockMessage("seen")})()]


class TestDirectMultimodal:
    async def test_sends_image_parts_to_litellm(self, tmp_path):
        img = tmp_path / "dot.png"
        img.write_bytes(_TINY_PNG)
        examples = [
            EvalExample(input="what color?", expected="red", extra={"images": ["dot.png"]}),
        ]
        captured: dict = {}

        async def _capture(*args, **kwargs):
            captured.update(kwargs)
            return _MockResponse()

        with patch("llmci.targets.direct.litellm.acompletion", side_effect=_capture):
            results = await run_direct_target(
                provider="openai",
                model="gpt-4o",
                prompt_template="{input}",
                examples=examples,
                media_base=tmp_path,
            )

        assert results[0].output == "seen"
        content = captured["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"

    async def test_cache_key_differs_by_image(self, tmp_path):
        img_a = tmp_path / "a.png"
        img_b = tmp_path / "b.png"
        img_a.write_bytes(_TINY_PNG)
        img_b.write_bytes(_TINY_PNG + b"x")

        counter = {"n": 0}

        async def _count(**kwargs):
            counter["n"] += 1
            return _MockResponse()

        cache = ResponseCache(tmp_path / "cache", enabled=True)
        with patch("llmci.targets.direct.litellm.acompletion", side_effect=_count):
            await run_direct_target(
                provider="openai", model="gpt-4o", prompt_template="{input}",
                examples=[EvalExample(input="q", expected="", extra={"images": ["a.png"]})],
                cache=cache, media_base=tmp_path,
            )
            await run_direct_target(
                provider="openai", model="gpt-4o", prompt_template="{input}",
                examples=[EvalExample(input="q", expected="", extra={"images": ["b.png"]})],
                cache=cache, media_base=tmp_path,
            )

        assert counter["n"] == 2

    def test_media_cache_params(self):
        ex = EvalExample(
            input="q", expected="", extra={"images": ["a.png"], "audio": ["b.wav"]},
        )
        assert media_cache_params(ex) == {"images": ["a.png"], "audio": ["b.wav"]}
