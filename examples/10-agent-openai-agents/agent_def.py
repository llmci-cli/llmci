#!/usr/bin/env python3
"""OpenAI Agents SDK agent for Scaffold evals (real API path)."""

from __future__ import annotations

from agents import Agent, function_tool

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


@function_tool
def get_weather(city: str) -> str:
    """Look up weather for a city."""
    for name, forecast in TOOL_DB["get_weather"].items():
        if name.lower() in city.lower():
            return forecast
    return "Weather data unavailable."


@function_tool
def search_docs(query: str) -> str:
    """Search support documentation."""
    query_lower = query.lower()
    for topic, answer in TOOL_DB["search_docs"].items():
        if topic.split()[0] in query_lower or topic in query_lower:
            return answer
    return "No documentation found."


@function_tool
def check_inventory(item: str) -> str:
    """Check product inventory."""
    item_lower = item.lower()
    for product, status in TOOL_DB["check_inventory"].items():
        if product in item_lower:
            return status
    return "Inventory status unknown."


def build_agent() -> Agent:
    return Agent(
        name="Scaffold Demo Agent",
        instructions=(
            "You are a helpful assistant. Use tools to answer questions. "
            "Always call the appropriate tool before responding."
        ),
        tools=[get_weather, search_docs, check_inventory],
    )
