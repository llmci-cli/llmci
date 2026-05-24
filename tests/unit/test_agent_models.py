"""Tests for agent data models."""

import pytest
from pydantic import ValidationError

from llmci.models import (
    AgentConstraints,
    AgentExpected,
    AgentScenario,
    AgentTrace,
    AgentTurn,
    TraceStep,
)


class TestAgentScenario:
    def test_single_turn(self):
        s = AgentScenario(
            input={"query": "hello"},
            expected=AgentExpected(outcome="greeting"),
        )
        assert not s.is_multi_turn

    def test_multi_turn(self):
        s = AgentScenario(
            turns=[
                AgentTurn(
                    user_message="hi",
                    expected=AgentExpected(outcome="greeting"),
                ),
                AgentTurn(
                    user_message="bye",
                    expected=AgentExpected(outcome="farewell"),
                ),
            ],
        )
        assert s.is_multi_turn

    def test_neither_raises(self):
        with pytest.raises(ValidationError, match="single-turn"):
            AgentScenario()

    def test_both_raises(self):
        with pytest.raises(ValidationError, match="single-turn.*multi-turn"):
            AgentScenario(
                input="x",
                expected=AgentExpected(outcome="y"),
                turns=[
                    AgentTurn(
                        user_message="z",
                        expected=AgentExpected(outcome="w"),
                    )
                ],
            )

    def test_string_input(self):
        s = AgentScenario(
            input="hello world",
            expected=AgentExpected(outcome="response"),
        )
        assert s.input == "hello world"

    def test_constraints_on_expected(self):
        s = AgentScenario(
            input="test",
            expected=AgentExpected(
                outcome="result",
                constraints=AgentConstraints(
                    max_tool_calls=3,
                    required_tools=["search"],
                    forbidden_tools=["delete"],
                ),
            ),
        )
        assert s.expected.constraints.max_tool_calls == 3
        assert s.expected.constraints.required_tools == ["search"]


class TestAgentTrace:
    def test_defaults(self):
        t = AgentTrace()
        assert t.final_output is None
        assert t.trace == []
        assert t.total_tool_calls == 0
        assert t.error is None

    def test_with_steps(self):
        t = AgentTrace(
            final_output="done",
            trace=[
                TraceStep(step=1, type="tool_call", tool="search", tokens=10),
                TraceStep(step=2, type="response", content="result", tokens=20),
            ],
            total_tool_calls=1,
            total_tokens=30,
        )
        assert t.total_tool_calls == 1
        assert len(t.trace) == 2

    def test_with_error(self):
        t = AgentTrace(error="something failed")
        assert t.error == "something failed"
