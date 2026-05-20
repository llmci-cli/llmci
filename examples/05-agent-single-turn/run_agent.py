#!/usr/bin/env python3
"""Mock single-turn agent for the Scaffold demo.

Simulates a tool-using agent that looks up information and responds.
Reads input JSON, writes a trace JSON with tool calls and final output.

Usage: python3 run_agent.py --input {input_file} --output {output_file}
"""

import argparse
import json

TOOL_DB = {
    "get_weather": {"New York": "72°F sunny", "London": "58°F cloudy", "Tokyo": "65°F rainy"},
    "search_docs": {
        "refund policy": "Refunds within 30 days of purchase.",
        "shipping": "Free shipping on orders over $50.",
        "returns": "Returns accepted within 14 days.",
    },
    "check_inventory": {
        "laptop": "In stock (42 units)",
        "headphones": "Out of stock",
        "keyboard": "In stock (15 units)",
    },
}


def run_agent(input_data: dict) -> dict:
    """Simulate an agent that uses tools to answer a query."""
    query = input_data.get("query", input_data.get("input", ""))
    if isinstance(query, dict):
        query = str(query)

    query_lower = query.lower()
    trace = []
    step = 0

    if "weather" in query_lower:
        for city in TOOL_DB["get_weather"]:
            if city.lower() in query_lower:
                step += 1
                trace.append({
                    "step": step, "type": "tool_call",
                    "tool": "get_weather",
                    "args": {"city": city},
                    "content": TOOL_DB["get_weather"][city],
                    "tokens": 25,
                })
                break

    elif "refund" in query_lower or "shipping" in query_lower or "return" in query_lower:
        for topic, answer in TOOL_DB["search_docs"].items():
            if topic.split()[0] in query_lower:
                step += 1
                trace.append({
                    "step": step, "type": "tool_call",
                    "tool": "search_docs",
                    "args": {"query": topic},
                    "content": answer,
                    "tokens": 30,
                })
                break

    elif "stock" in query_lower or "inventory" in query_lower or "available" in query_lower:
        for item in TOOL_DB["check_inventory"]:
            if item in query_lower:
                step += 1
                trace.append({
                    "step": step, "type": "tool_call",
                    "tool": "check_inventory",
                    "args": {"item": item},
                    "content": TOOL_DB["check_inventory"][item],
                    "tokens": 20,
                })
                break

    step += 1
    if trace:
        tool_result = trace[-1]["content"]
        final_output = f"Based on my research: {tool_result}"
    else:
        final_output = "I don't have specific tools to answer that question."

    trace.append({
        "step": step, "type": "response",
        "content": final_output, "tokens": 40,
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
