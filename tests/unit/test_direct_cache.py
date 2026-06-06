"""Tests that the direct target uses the response cache."""

from unittest.mock import patch

from llmci.cache import ResponseCache
from llmci.models import EvalExample
from llmci.targets.direct import run_direct_target


class _MockMessage:
    def __init__(self, content: str):
        self.content = content


class _MockChoice:
    def __init__(self, content: str):
        self.message = _MockMessage(content)


class _MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockResponse:
    def __init__(self, content: str, usage=None, cost=None):
        self.choices = [_MockChoice(content)]
        if usage is not None:
            self.usage = usage
        if cost is not None:
            self._hidden_params = {"response_cost": cost}


def _counting_acompletion(content: str, counter: dict):
    async def _mock(*args, **kwargs):
        counter["calls"] += 1
        return _MockResponse(content)

    return _mock


async def test_second_run_is_cache_hit(tmp_path):
    counter = {"calls": 0}
    cache = ResponseCache(cache_dir=tmp_path)
    examples = [EvalExample(input="ping", expected="pong")]

    with patch(
        "llmci.targets.direct.litellm.acompletion",
        side_effect=_counting_acompletion("pong", counter),
    ):
        first = await run_direct_target(
            provider="openai", model="gpt-4o", prompt_template="Q: {input}",
            examples=examples, cache=cache,
        )
        second = await run_direct_target(
            provider="openai", model="gpt-4o", prompt_template="Q: {input}",
            examples=examples, cache=cache,
        )

    assert first[0].output == "pong"
    assert second[0].output == "pong"
    # The provider was hit once; the second run was served from cache.
    assert counter["calls"] == 1
    assert cache.hits == 1


async def test_no_cache_always_calls_provider(tmp_path):
    counter = {"calls": 0}
    cache = ResponseCache(cache_dir=tmp_path, enabled=False)
    examples = [EvalExample(input="ping", expected="pong")]

    with patch(
        "llmci.targets.direct.litellm.acompletion",
        side_effect=_counting_acompletion("pong", counter),
    ):
        await run_direct_target(
            provider="openai", model="gpt-4o", prompt_template="{input}",
            examples=examples, cache=cache,
        )
        await run_direct_target(
            provider="openai", model="gpt-4o", prompt_template="{input}",
            examples=examples, cache=cache,
        )

    assert counter["calls"] == 2


async def test_usage_and_cost_captured_and_cached(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path)
    examples = [EvalExample(input="ping", expected="pong")]

    async def _mock(*args, **kwargs):
        return _MockResponse("pong", usage=_MockUsage(120, 30), cost=0.0042)

    with patch("llmci.targets.direct.litellm.acompletion", side_effect=_mock):
        first = await run_direct_target(
            provider="openai", model="gpt-4o", prompt_template="{input}",
            examples=examples, cache=cache,
        )

    assert first[0].tokens_in == 120
    assert first[0].tokens_out == 30
    assert first[0].cost == 0.0042

    # Replayed from cache with the same usage/cost.
    replay = ResponseCache(cache_dir=tmp_path)
    with patch("llmci.targets.direct.litellm.acompletion", side_effect=Exception("no call")):
        second = await run_direct_target(
            provider="openai", model="gpt-4o", prompt_template="{input}",
            examples=examples, cache=replay,
        )

    assert second[0].tokens_in == 120
    assert second[0].tokens_out == 30
    assert second[0].cost == 0.0042
    assert replay.hits == 1


async def test_errors_are_not_cached(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path)
    examples = [EvalExample(input="ping", expected="pong")]

    with patch(
        "llmci.targets.direct.litellm.acompletion",
        side_effect=Exception("boom"),
    ):
        results = await run_direct_target(
            provider="openai", model="gpt-4o", prompt_template="{input}",
            examples=examples, retries=0, cache=cache,
        )

    assert results[0].error is not None
    assert not list(tmp_path.glob("*.json"))
