#!/usr/bin/env python3
"""Mock ticket classifier for the integrated CI-gate example.

Usage: python3 run_classifier.py --input {input_file} --output {output_file}

Returns a category label plus simulated token usage and cost so llmci can gate on
``cost_mean`` / ``tokens_in_mean`` with ``max_regression`` vs a baseline — the same
metrics you'd track on a real direct-API target.

Set ``INFLATE_COST = True`` to simulate a prompt change that bloated token usage and
watch the cost regression gate fail.
"""

import argparse
import json

INFLATE_COST = False

KEYWORDS = {
    "hardware": ["printer", "monitor", "keyboard", "webcam", "hard drive", "pixel", "wifi", "cable", "mouse", "screen"],
    "billing": ["refund", "charged", "invoice", "subscription", "upgrade", "cancel", "payment", "billing", "plan", "price"],
    "account": ["password", "email", "export", "unsubscribe", "two-factor", "login", "sign", "account", "profile", "authentication"],
    "software": ["crash", "app", "website", "loading", "error", "search", "notification", "bug", "update", "slow"],
}


def classify(text: str) -> str:
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in KEYWORDS.items()}
    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=lambda k: scores.get(k, 0))


def usage_for(text: str) -> dict:
    """Simulate token/cost accounting proportional to input size."""
    tokens_in = max(40, len(text) * 3)
    tokens_out = 8
    if INFLATE_COST:
        tokens_in *= 3
        tokens_out *= 2
    cost = tokens_in * 0.000002 + tokens_out * 0.000006
    return {"usage": {"tokens_in": tokens_in, "tokens_out": tokens_out}, "cost": round(cost, 6)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    payload = {"output": classify(data["input"]), **usage_for(data["input"])}
    json.dump(payload, open(args.output, "w"))


if __name__ == "__main__":
    main()
