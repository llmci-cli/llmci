#!/usr/bin/env python3
"""Mock multi-turn agent for the Scaffold demo.

Simulates a conversational agent that maintains context across turns.
Reads input JSON (with history), writes a trace JSON.

Usage: python3 run_agent.py --input {input_file} --output {output_file}
"""

import argparse
import json


def run_agent(input_data: dict) -> dict:
    """Simulate a multi-turn conversational agent."""
    user_message = input_data.get("user_message", "")
    history = input_data.get("history", [])
    turn_index = input_data.get("turn_index", 0)

    trace = []
    step = 0

    context_from_history = ""
    for msg in history:
        if msg.get("role") == "assistant":
            context_from_history = msg.get("content", "")

    msg_lower = user_message.lower()

    if "order" in msg_lower and "status" in msg_lower:
        step += 1
        trace.append({
            "step": step, "type": "tool_call",
            "tool": "lookup_order",
            "args": {"query": user_message},
            "content": "Order #1234: Shipped, ETA 2 days",
            "tokens": 30,
        })
        final_output = "Your order #1234 has been shipped and should arrive in 2 days."

    elif "cancel" in msg_lower:
        step += 1
        trace.append({
            "step": step, "type": "tool_call",
            "tool": "cancel_order",
            "args": {"order_id": "1234"},
            "content": "Order #1234 cancelled successfully",
            "tokens": 25,
        })
        final_output = "I've cancelled your order #1234. You'll receive a refund within 3-5 days."

    elif "thank" in msg_lower:
        final_output = "You're welcome! Is there anything else I can help with?"

    else:
        step += 1
        trace.append({
            "step": step, "type": "tool_call",
            "tool": "search_kb",
            "args": {"query": user_message},
            "content": "General support information",
            "tokens": 20,
        })
        final_output = f"I found some information about: {user_message}"

    step += 1
    trace.append({
        "step": step, "type": "response",
        "content": final_output, "tokens": 35,
    })

    total_tokens = sum(t.get("tokens", 0) for t in trace)
    tool_calls = sum(1 for t in trace if t["type"] == "tool_call")

    return {
        "final_output": final_output,
        "trace": trace,
        "total_tool_calls": tool_calls,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    result = run_agent(data)
    json.dump(result, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
