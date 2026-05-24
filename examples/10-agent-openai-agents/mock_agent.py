#!/usr/bin/env python3
"""Mock agent path using TraceBuilder (CI / no API key).

Same tool names as agent_def.py so constraint evals match the dataset.
"""

from __future__ import annotations

from scaffold.trace import TraceBuilder

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


def run_mock(input_data: dict) -> dict:
    query = input_data.get("query", input_data.get("input", ""))
    if isinstance(query, dict):
        query = str(query.get("query", query))
    query_lower = str(query).lower()

    tb = TraceBuilder()

    if "weather" in query_lower:
        for city in TOOL_DB["get_weather"]:
            if city.lower() in query_lower:
                forecast = TOOL_DB["get_weather"][city]
                tb.tool("get_weather", {"city": city}, result=forecast, tokens=25)
                tb.response(f"Based on my research: {forecast}", tokens=40)
                return tb.to_dict()

    if any(k in query_lower for k in ("refund", "shipping", "return")):
        for topic, answer in TOOL_DB["search_docs"].items():
            if topic.split()[0] in query_lower:
                tb.tool("search_docs", {"query": topic}, result=answer, tokens=30)
                tb.response(f"Based on my research: {answer}", tokens=40)
                return tb.to_dict()

    if any(k in query_lower for k in ("stock", "inventory", "available")):
        for item in TOOL_DB["check_inventory"]:
            if item in query_lower:
                status = TOOL_DB["check_inventory"][item]
                tb.tool("check_inventory", {"item": item}, result=status, tokens=20)
                tb.response(f"Based on my research: {status}", tokens=40)
                return tb.to_dict()

    tb.response("I don't have specific tools to answer that question.", tokens=40)
    return tb.to_dict()
