"""Tests for the composite judge system."""


import pytest

from scaffold.judges.composite import (
    CompositeAgentJudge,
    ConstraintJudge,
    _parse_judge_response,
)
from scaffold.models import (
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
