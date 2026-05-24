"""Tests for the exact match judge."""

import pytest

from llmci.judges.exact_match import ExactMatchJudge


@pytest.fixture
def judge():
    return ExactMatchJudge()


@pytest.fixture
def judge_ci():
    return ExactMatchJudge(case_sensitive=False)


class TestExactMatchJudge:
    @pytest.mark.asyncio
    async def test_exact_match(self, judge):
        result = await judge.evaluate_single("q", "hello", "hello")
        assert result.score == 1.0
        assert result.reason is None

    @pytest.mark.asyncio
    async def test_mismatch(self, judge):
        result = await judge.evaluate_single("q", "hello", "world")
        assert result.score == 0.0
        assert "Expected" in result.reason

    @pytest.mark.asyncio
    async def test_whitespace_stripping(self, judge):
        result = await judge.evaluate_single("q", "hello", "  hello  ")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_case_sensitive_by_default(self, judge):
        result = await judge.evaluate_single("q", "Hello", "hello")
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self, judge_ci):
        result = await judge_ci.evaluate_single("q", "Hello", "hello")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_strings(self, judge):
        result = await judge.evaluate_single("q", "", "")
        assert result.score == 1.0
