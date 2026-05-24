"""Tests for TraceBuilder."""

import json

from scaffold.trace import TraceBuilder


class TestTraceBuilder:
    def test_tool_and_response(self):
        tb = TraceBuilder()
        tb.tool("get_weather", {"city": "NYC"}, result="72°F", tokens=10)
        tb.response("It's 72°F in NYC.", tokens=20)

        out = tb.to_dict()
        assert out["final_output"] == "It's 72°F in NYC."
        assert out["total_tool_calls"] == 1
        assert out["total_tokens"] == 30
        assert len(out["trace"]) == 2
        assert out["trace"][0]["type"] == "tool_call"
        assert out["trace"][0]["tool"] == "get_weather"

    def test_to_json_roundtrip(self):
        tb = TraceBuilder()
        tb.response("Hello", tokens=5)
        parsed = json.loads(tb.to_json())
        assert parsed["final_output"] == "Hello"

    def test_infers_final_from_last_response(self):
        tb = TraceBuilder()
        tb.tool("search_docs", {"q": "refund"}, tokens=5)
        tb.response("Refunds in 30 days.", tokens=10)
        assert tb.final_output == "Refunds in 30 days."
