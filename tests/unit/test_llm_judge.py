"""Tests for LLM-as-judge with mocked litellm."""

from unittest.mock import patch

import pytest

from scaffold.judges.llm_judge import LLMJudge


class MockChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})()


class MockResponse:
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


def _mock_acompletion(responses: list[str]):
    """Create a mock that returns responses in sequence."""
    idx = {"i": 0}

    async def _mock(*args, **kwargs):
        content = responses[idx["i"] % len(responses)]
        idx["i"] += 1
        return MockResponse(content)

    return _mock


class TestLLMJudge:
    @pytest.mark.asyncio
    @patch("scaffold.judges.llm_judge.litellm.acompletion")
    async def test_all_criteria_pass(self, mock_llm):
        mock_llm.side_effect = _mock_acompletion([
            '{"passed": true, "reasoning": "Looks good"}',
        ])

        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric=[
                {"id": "accuracy", "prompt": "Is it accurate?"},
                {"id": "tone", "prompt": "Is the tone appropriate?"},
            ],
            use_cache=False,
        )

        result = await judge.evaluate_single("input", "expected", "actual")
        assert result.score == 1.0
        assert result.reason is None
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    @patch("scaffold.judges.llm_judge.litellm.acompletion")
    async def test_one_criterion_fails(self, mock_llm):
        mock_llm.side_effect = _mock_acompletion([
            '{"passed": true, "reasoning": "Good"}',
            '{"passed": false, "reasoning": "Too verbose"}',
        ])

        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric=[
                {"id": "accuracy", "prompt": "Is it accurate?"},
                {"id": "conciseness", "prompt": "Is it concise?"},
            ],
            use_cache=False,
        )

        result = await judge.evaluate_single("input", "expected", "actual")
        assert result.score == 0.5
        assert "conciseness" in result.reason

    @pytest.mark.asyncio
    @patch("scaffold.judges.llm_judge.litellm.acompletion")
    async def test_string_rubric(self, mock_llm):
        mock_llm.side_effect = _mock_acompletion([
            '{"passed": true, "reasoning": "Yes"}',
        ])

        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric="Is the output helpful?",
            use_cache=False,
        )

        result = await judge.evaluate_single("q", "ref", "answer")
        assert result.score == 1.0
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    @patch("scaffold.judges.llm_judge.litellm.acompletion")
    async def test_malformed_response_fallback(self, mock_llm):
        mock_llm.side_effect = _mock_acompletion([
            "This is not JSON at all",
        ])

        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric=[{"id": "test", "prompt": "Test criterion"}],
            use_cache=False,
        )

        result = await judge.evaluate_single("q", "ref", "answer")
        assert result.score == 0.0
        assert "parse" in result.reason.lower() or "Could not" in result.reason

    @pytest.mark.asyncio
    @patch("scaffold.judges.llm_judge.litellm.acompletion")
    async def test_llm_error_handled(self, mock_llm):
        mock_llm.side_effect = Exception("API rate limit exceeded")

        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric=[{"id": "test", "prompt": "Test"}],
            use_cache=False,
        )

        result = await judge.evaluate_single("q", "ref", "answer")
        assert result.score == 0.0
        assert "error" in result.reason.lower()

    @pytest.mark.asyncio
    @patch("scaffold.judges.llm_judge.litellm.acompletion")
    async def test_code_fence_stripped(self, mock_llm):
        mock_llm.side_effect = _mock_acompletion([
            '```json\n{"passed": true, "reasoning": "OK"}\n```',
        ])

        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric=[{"id": "test", "prompt": "Test"}],
            use_cache=False,
        )

        result = await judge.evaluate_single("q", "ref", "answer")
        assert result.score == 1.0

    @pytest.mark.asyncio
    @patch("scaffold.judges.llm_judge.litellm.acompletion")
    async def test_temperature_zero(self, mock_llm):
        mock_llm.side_effect = _mock_acompletion([
            '{"passed": true, "reasoning": "OK"}',
        ])

        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric=[{"id": "test", "prompt": "Test"}],
            use_cache=False,
        )

        await judge.evaluate_single("q", "ref", "answer")
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_cache_key_deterministic(self):
        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric=[{"id": "test", "prompt": "Test"}],
            use_cache=False,
        )
        k1 = judge._cache_key("c", "i", "e", "a")
        k2 = judge._cache_key("c", "i", "e", "a")
        assert k1 == k2

    def test_cache_key_varies(self):
        judge = LLMJudge(
            model="openai/gpt-4o",
            rubric=[{"id": "test", "prompt": "Test"}],
            use_cache=False,
        )
        k1 = judge._cache_key("c1", "i", "e", "a")
        k2 = judge._cache_key("c2", "i", "e", "a")
        assert k1 != k2
