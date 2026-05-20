#!/usr/bin/env python3
"""Simple deterministic ticket classifier for the Scaffold demo.

Reads an input JSON file, classifies the ticket, writes output JSON.
Usage: python run_prompt.py --input {input_file} --output {output_file}
"""

import json
import sys

KEYWORDS = {
    "hardware": [
        "printer", "monitor", "keyboard",
        "pixel", "cable", "screen",
    ],
    "billing": [
        "refund", "invoice", "subscription",
        "cancel", "billing",
    ],
    "account": [
        "password", "email", "export", "unsubscribe", "two-factor",
        "login", "sign", "account", "profile", "authentication",
    ],
    "software": [
        "crash", "app", "website", "loading", "error",
        "search", "notification", "bug", "update", "slow",
    ],
}


def classify(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for category, keywords in KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in text_lower)

    if max(scores.values()) == 0:
        return "general"

    return max(scores, key=lambda k: scores[k])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    result = classify(data["input"])
    json.dump({"output": result}, open(args.output, "w"))


if __name__ == "__main__":
    main()
