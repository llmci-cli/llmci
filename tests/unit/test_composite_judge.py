"""Tests for the composite judge system."""

from unittest.mock import patch

import pytest

from llmci.cache import ResponseCache
from llmci.judges.composite import (
    CompositeAgentJudge,
    ConstraintJudge,
    OutcomeJudge,
    _parse_judge_response,
)
from llmci.models import (
    AgentConstraints,
    AgentExpected,
    AgentScenario,
    AgentTrace,
    TraceStep,
)


class TestConstraintJudge:
    def test_all_pass(self):
        trace = AgentTrace(
            total_tool_calls=2,
            total_tokens=100,
            trace=[
                TraceStep(step=1, type="tool_call", tool="search", tokens=50),
                TraceStep(step=2, type="tool_call", tool="lookup", tokens=50),
            ],
        )
        constraints = AgentConstraints(
            max_tool_calls=5,
            max_tokens=200,
            required_tools=["search"],
        )
        result = ConstraintJudge().evaluate(trace, constraints)
        assert result.score == 1.0
        assert "All constraints satisfied" in result.reason

    def test_tool_call_exceeded(self):
        trace = AgentTrace(total_tool_calls=5, trace=[])
        constraints = AgentConstraints(max_tool_calls=3)
        result = ConstraintJudge().evaluate(trace, constraints)
        assert result.score == 0.0
        assert "tool_calls" in result.reason

    def test_token_exceeded(self):
        trace = AgentTrace(total_tokens=500, trace=[])
        constraints = AgentConstraints(max_tokens=100)
        result = ConstraintJudge().evaluate(trace, constraints)
        assert result.score == 0.0

    def test_required_tool_missing(self):
        trace = AgentTrace(
            trace=[
                TraceStep(step=1, type="tool_call", tool="search"),
            ],
        )
        constraints = AgentConstraints(required_tools=["search", "lookup"])
        result = ConstraintJudge().evaluate(trace, constraints)
        assert result.score == 0.5

    def test_forbidden_tool_present(self):
        trace = AgentTrace(
            trace=[
                TraceStep(step=1, type="tool_call", tool="delete"),
            ],
        )
        constraints = AgentConstraints(forbidden_tools=["delete"])
        result = ConstraintJudge().evaluate(trace, constraints)
        assert result.score == 0.0
        assert "forbidden" in result.reason

    def test_no_constraints(self):
        trace = AgentTrace()
        constraints = AgentConstraints()
        result = ConstraintJudge().evaluate(trace, constraints)
        assert result.score == 1.0
        assert "No constraints" in result.reason

    def test_partial_pass(self):
        trace = AgentTrace(
            total_tool_calls=5,
            total_tokens=50,
            trace=[
                TraceStep(step=1, type="tool_call", tool="search"),
            ],
        )
        constraints = AgentConstraints(
            max_tool_calls=3,
            max_tokens=100,
            required_tools=["search"],
        )
        result = ConstraintJudge().evaluate(trace, constraints)
        assert 0.0 < result.score < 1.0


class TestParseJudgeResponse:
    def test_valid_json(self):
        result = _parse_judge_response('{"score": 0.8, "reason": "good"}')
        assert result.score == 0.8
        assert result.reason == "good"

    def test_code_fenced_json(self):
        result = _parse_judge_response('```json\n{"score": 0.9, "reason": "great"}\n```')
        assert result.score == 0.9

    def test_invalid_json(self):
        result = _parse_judge_response("not json at all")
        assert result.score == 0.0
        assert "Could not parse" in result.reason

    def test_score_clamped(self):
        result = _parse_judge_response('{"score": 1.5, "reason": "over"}')
        assert result.score == 1.0

    def test_score_clamped_negative(self):
        result = _parse_judge_response('{"score": -0.5, "reason": "under"}')
        assert result.score == 0.0


class TestCompositeAgentJudge:
    @pytest.mark.asyncio
    async def test_constraint_only(self):
        judge = CompositeAgentJudge(
            criteria=[{"name": "constraints", "type": "constraint", "weight": 1.0}],
        )
        scenario = AgentScenario(
            input="test",
            expected=AgentExpected(
                outcome="result",
                constraints=AgentConstraints(max_tool_calls=5),
            ),
        )
        trace = AgentTrace(total_tool_calls=2, trace=[])
        result = await judge.evaluate_scenario(scenario, trace)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_weighted_combination(self):
        judge = CompositeAgentJudge(
            criteria=[
                {"name": "c1", "type": "constraint", "weight": 1.0},
                {"name": "c2", "type": "constraint", "weight": 1.0},
            ],
        )
        scenario = AgentScenario(
            input="test",
            expected=AgentExpected(
                outcome="result",
                constraints=AgentConstraints(
                    max_tool_calls=5,
                    max_tokens=100,
                ),
            ),
        )
        trace = AgentTrace(total_tool_calls=2, total_tokens=50, trace=[])
        result = await judge.evaluate_scenario(scenario, trace)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_unknown_criterion_type(self):
        judge = CompositeAgentJudge(
            criteria=[{"name": "unknown", "type": "magic", "weight": 1.0}],
        )
        scenario = AgentScenario(
            input="test",
            expected=AgentExpected(outcome="result"),
        )
        trace = AgentTrace()
        result = await judge.evaluate_scenario(scenario, trace)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_no_criteria(self):
        judge = CompositeAgentJudge(criteria=[])
        scenario = AgentScenario(
            input="test",
            expected=AgentExpected(outcome="result"),
        )
        trace = AgentTrace()
        result = await judge.evaluate_scenario(scenario, trace)
        assert result.score == 0.0


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


class TestCompositeJudgeCache:
    @pytest.mark.asyncio
    async def test_outcome_judge_reuses_cache(self, tmp_path):
        counter = {"n": 0}
        cache = ResponseCache(tmp_path / "judges", enabled=True)
        judge = OutcomeJudge(model="gpt-4o-mini")
        scenario = AgentScenario(
            input="test",
            expected=AgentExpected(outcome="done"),
        )
        trace = AgentTrace(final_output="hello")

        with patch(
            "llmci.judges.llm_cache.litellm.acompletion",
            side_effect=_counting('{"score": 0.9, "reason": "cached"}', counter),
        ):
            first = await judge.evaluate(scenario, trace, cache=cache)
            second = await judge.evaluate(scenario, trace, cache=cache)

        assert first.score == pytest.approx(0.9)
        assert second.score == pytest.approx(0.9)
        assert counter["n"] == 1
        assert cache.hits == 1

    @pytest.mark.asyncio
    async def test_composite_passes_attached_cache(self, tmp_path):
        counter = {"n": 0}
        cache = ResponseCache(tmp_path / "judges", enabled=True)
        judge = CompositeAgentJudge(
            criteria=[{"name": "outcome", "type": "outcome", "weight": 1.0}],
        )
        judge.set_judge_cache(cache)
        scenario = AgentScenario(
            input="test",
            expected=AgentExpected(outcome="done"),
        )
        trace = AgentTrace(final_output="hello")

        with patch(
            "llmci.judges.llm_cache.litellm.acompletion",
            side_effect=_counting('{"score": 1.0, "reason": "ok"}', counter),
        ):
            await judge.evaluate_scenario(scenario, trace)
            await judge.evaluate_scenario(scenario, trace)

        assert counter["n"] == 1
