#!/usr/bin/env python3
"""A deterministic keyword ticket classifier.

Usage: python3 run_classifier.py --input {input_file} --output {output_file}

Categories: billing, technical, account.
"""

import argparse
import json

RULES = {
    "billing": ("charge", "charged", "bill", "subscription", "renew", "refund", "invoice"),
    "technical": ("crash", "error", "bug", "export", "button", "broken", "slow"),
    "account": ("log in", "log into", "login", "password", "reset", "account", "sign in"),
}


def classify(text: str) -> str:
    lowered = text.lower()
    for category, keywords in RULES.items():
        if any(kw in lowered for kw in keywords):
            return category
    return "technical"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    with open(args.output, "w") as f:
        json.dump({"output": classify(data["input"])}, f)


if __name__ == "__main__":
    main()
