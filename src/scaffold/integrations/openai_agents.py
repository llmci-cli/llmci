"""Convert OpenAI Agents SDK runs into Scaffold agent eval output."""

from __future__ import annotations

import json
from typing import Any

from scaffold.trace import TraceBuilder


def scaffold_input_to_agent_input(input_data: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Map Scaffold agent eval input to OpenAI Agents SDK runner input."""
    if "user_message" in input_data:
        messages: list[dict[str, Any]] = []
        for msg in input_data.get("history", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": input_data["user_message"]})
        return messages

    if "query" in input_data:
        query = input_data["query"]
        return query if isinstance(query, str) else json.dumps(query)

    if "input" in input_data:
        raw = input_data["input"]
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict) and "query" in raw:
            return str(raw["query"])
        return json.dumps(raw)

    return json.dumps(input_data)


def scaffold_output_from_run_result(result: Any) -> dict[str, Any]:
    """Convert an OpenAI Agents ``RunResult`` into Scaffold trace JSON."""
    builder = TraceBuilder()

    new_items = getattr(result, "new_items", None) or []
    pending_calls: dict[str, dict[str, Any]] = {}

    for item in new_items:
        item_type = getattr(item, "type", None)

        if item_type == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            name, args, call_id = _extract_tool_call(raw)
            pending_calls[call_id or name] = {"name": name, "args": args}
            builder.tool(name, args)

        elif item_type == "tool_call_output_item":
            raw = getattr(item, "raw_item", None)
            output = getattr(item, "output", None)
            call_id, name = _extract_tool_output_ref(raw)
            key = call_id or name
            if key and key in pending_calls:
                # Attach result text to the last matching tool step if empty
                _attach_tool_result(builder, pending_calls[key]["name"], _stringify(output))

    final_output = getattr(result, "final_output", None)
    if final_output is not None:
        text = _stringify(final_output)
    else:
        text = _extract_message_text(new_items)

    if text:
        builder.response(text)

    return builder.to_dict()


async def run_for_scaffold(agent: Any, input_data: dict[str, Any], **run_kwargs: Any) -> dict[str, Any]:
    """Run an OpenAI Agents agent and return Scaffold-compatible output."""
    from agents import Runner

    agent_input = scaffold_input_to_agent_input(input_data)
    result = await Runner.run(agent, agent_input, **run_kwargs)
    return scaffold_output_from_run_result(result)


def run_for_scaffold_sync(agent: Any, input_data: dict[str, Any], **run_kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper around :func:`run_for_scaffold`."""
    from agents import Runner

    agent_input = scaffold_input_to_agent_input(input_data)
    result = Runner.run_sync(agent, agent_input, **run_kwargs)
    return scaffold_output_from_run_result(result)


def _extract_tool_call(raw: Any) -> tuple[str, dict[str, Any], str | None]:
    if raw is None:
        return "unknown_tool", {}, None

    name = getattr(raw, "name", None) or "unknown_tool"
    call_id = getattr(raw, "call_id", None)

    args_raw = getattr(raw, "arguments", None)
    args: dict[str, Any] = {}
    if isinstance(args_raw, str) and args_raw.strip():
        try:
            parsed = json.loads(args_raw)
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            args = {"raw": args_raw}
    elif isinstance(args_raw, dict):
        args = args_raw

    return str(name), args, call_id


def _extract_tool_output_ref(raw: Any) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    call_id = getattr(raw, "call_id", None)
    name = getattr(raw, "name", None)
    return call_id, name


def _attach_tool_result(builder: TraceBuilder, tool_name: str, result: str) -> None:
    for step in reversed(builder.trace):
        if step.get("type") == "tool_call" and step.get("tool") == tool_name:
            if not step.get("content"):
                step["content"] = result
            return


def _extract_message_text(items: list[Any]) -> str:
    try:
        from agents.items import ItemHelpers

        return ItemHelpers.extract_last_text(items)
    except Exception:
        pass

    for item in reversed(items):
        if getattr(item, "type", None) != "message_output_item":
            continue
        raw = getattr(item, "raw_item", None)
        content = getattr(raw, "content", None) if raw is not None else None
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
                elif isinstance(block, dict) and block.get("text"):
                    parts.append(str(block["text"]))
            if parts:
                return "".join(parts)
    return ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump())
    return str(value)
