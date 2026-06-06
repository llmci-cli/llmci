"""Tests for the shared LLM judge-call cache."""

from unittest.mock import patch

import pytest

from llmci.cache import ResponseCache
from llmci.judges import llm_cache


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]


def _counting(content, counter):
    async def _m(**kwargs):
        counter["n"] += 1
        return _Resp(content)

    return _m


class TestJudgeCacheFrom:
    def test_none_when_target_cache_absent(self):
        assert llm_cache.judge_cache_from(None) is None

    def test_none_when_target_cache_disabled(self, tmp_path):
        disabled = ResponseCache(tmp_path, enabled=False)
        assert llm_cache.judge_cache_from(disabled) is None

    def test_mirrors_refresh_flag(self, tmp_path):
        target = ResponseCache(tmp_path, enabled=True, refresh=True)
        jc = llm_cache.judge_cache_from(target)
        assert jc is not None
        assert jc.refresh is True


class TestComplete:
    async def test_no_cache_calls_through(self):
        counter = {"n": 0}
        with patch("llmci.judges.llm_cache.litellm.acompletion",
                   side_effect=_counting("hi", counter)):
            out = await llm_cache.complete("m", "prompt", cache=None)
        assert out == "hi"
        assert counter["n"] == 1

    async def test_second_call_served_from_cache(self, tmp_path):
        counter = {"n": 0}
        cache = ResponseCache(tmp_path / "judges", enabled=True)
        with patch("llmci.judges.llm_cache.litellm.acompletion",
                   side_effect=_counting("answer", counter)):
            first = await llm_cache.complete("m", "same prompt", cache=cache)
            second = await llm_cache.complete("m", "same prompt", cache=cache)

        assert first == second == "answer"
        assert counter["n"] == 1  # second served from disk
        assert cache.hits == 1

    async def test_different_prompt_misses(self, tmp_path):
        counter = {"n": 0}
        cache = ResponseCache(tmp_path / "judges", enabled=True)
        with patch("llmci.judges.llm_cache.litellm.acompletion",
                   side_effect=_counting("x", counter)):
            await llm_cache.complete("m", "prompt A", cache=cache)
            await llm_cache.complete("m", "prompt B", cache=cache)
        assert counter["n"] == 2

    async def test_error_not_cached(self, tmp_path):
        cache = ResponseCache(tmp_path / "judges", enabled=True)

        async def _boom(**kwargs):
            raise RuntimeError("provider down")

        with patch("llmci.judges.llm_cache.litellm.acompletion", side_effect=_boom):
            with pytest.raises(RuntimeError):
                await llm_cache.complete("m", "p", cache=cache)
        # Nothing stored, so a later good call still goes live.
        assert cache.hits == 0
