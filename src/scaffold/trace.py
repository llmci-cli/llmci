"""Build Scaffold agent eval output (trace JSON) from framework runs."""

from __future__ import annotations

import json
from typing import Any


class TraceBuilder:
    """Fluent builder for Scaffold agent command-mode output.

    Example::

        tb = TraceBuilder()
        tb.tool("get_weather", {"city": "London"}, result="58°F cloudy", tokens=25)
        tb.response("It's 58°F and cloudy in London.")
        return tb.to_dict()
    """

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._step = 0
        self._final_output: str | None = None

    def tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        result: str | None = None,
        tokens: int | None = None,
    ) -> TraceBuilder:
        """Record a tool call step."""
        self._step += 1
        self._steps.append({
            "step": self._step,
            "type": "tool_call",
            "tool": name,
            "args": args or {},
            "content": result,
            "tokens": tokens,
        })
        return self

    def response(self, text: str, *, tokens: int | None = None) -> TraceBuilder:
        """Record the final assistant response."""
        self._final_output = text
        self._step += 1
        self._steps.append({
            "step": self._step,
            "type": "response",
            "content": text,
            "tokens": tokens,
        })
        return self

    @property
    def final_output(self) -> str | None:
        return self._final_output

    @property
    def trace(self) -> list[dict[str, Any]]:
        return list(self._steps)

    def to_dict(self) -> dict[str, Any]:
        """Return Scaffold agent output JSON."""
        final = self._final_output
        if final is None:
            for step in reversed(self._steps):
                if step.get("type") == "response" and step.get("content"):
                    final = str(step["content"])
                    break
            if final is None:
                final = ""

        tool_calls = sum(1 for s in self._steps if s.get("type") == "tool_call")
        total_tokens = sum(int(s["tokens"]) for s in self._steps if s.get("tokens"))

        return {
            "final_output": final,
            "trace": self._steps,
            "total_tool_calls": tool_calls,
            "total_tokens": total_tokens,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
