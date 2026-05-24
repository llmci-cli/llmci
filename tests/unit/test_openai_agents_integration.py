"""Tests for OpenAI Agents SDK integration helpers."""

from types import SimpleNamespace

from scaffold.integrations.openai_agents import (
    scaffold_input_to_agent_input,
    scaffold_output_from_run_result,
)


class TestScaffoldInputMapping:
    def test_single_turn_query(self):
        assert scaffold_input_to_agent_input({"query": "hello"}) == "hello"

    def test_multi_turn_messages(self):
        inp = {
            "user_message": "Cancel it",
            "history": [
                {"role": "user", "content": "Order status?"},
                {"role": "assistant", "content": "Shipped."},
            ],
        }
        messages = scaffold_input_to_agent_input(inp)
        assert isinstance(messages, list)
        assert messages[-1] == {"role": "user", "content": "Cancel it"}
        assert len(messages) == 3


class TestRunResultMapping:
    def test_maps_tool_call_and_message(self):
        tool_raw = SimpleNamespace(
            name="get_weather",
            arguments='{"city": "London"}',
            call_id="call_1",
        )
        tool_item = SimpleNamespace(type="tool_call_item", raw_item=tool_raw)
        msg_raw = SimpleNamespace(content="It's cloudy in London.")
        msg_item = SimpleNamespace(type="message_output_item", raw_item=msg_raw)
        result = SimpleNamespace(
            new_items=[tool_item, msg_item],
            final_output="It's cloudy in London.",
        )

        out = scaffold_output_from_run_result(result)
        assert out["final_output"] == "It's cloudy in London."
        assert out["total_tool_calls"] == 1
        assert out["trace"][0]["tool"] == "get_weather"
        assert out["trace"][0]["args"] == {"city": "London"}
