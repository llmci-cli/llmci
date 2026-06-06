#!/usr/bin/env python3
"""Mock support responder for the integrated CI-gate example.

Usage: python3 run_responder.py --input {input_file} --output {output_file}

Produces a short, policy-safe reply for each ticket. The ``safety`` judge's deterministic
``pii_leakage`` criterion scans the reply — no API key required.

Set ``LEAK_PII = True`` to echo fake customer data and watch the safety gate fail.
"""

import argparse
import json

LEAK_PII = False

SAFE_REPLIES = {
    "billing": "I can't access billing details here. Please open the secure billing portal to view charges or request a refund.",
    "account": "For account changes, use the account settings page or contact support through the verified channel — I can't share personal details in chat.",
    "hardware": "Try the troubleshooting guide in our help center. If the issue persists, open a hardware support ticket from your dashboard.",
    "software": "Please restart the app and check for updates. If errors continue, send a bug report from Settings → Help.",
    "general": "Thanks for reaching out. Could you share a bit more detail so we can route your request correctly?",
}

LEAKED = (
    "Sure — the account on file is jane.doe@example.com, phone 415-555-0199, "
    "SSN 123-45-6789."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    text = data["input"].lower()
    if LEAK_PII:
        reply = LEAKED
    elif any(w in text for w in ("refund", "charged", "invoice", "billing", "subscription")):
        reply = SAFE_REPLIES["billing"]
    elif any(w in text for w in ("password", "email", "account", "login", "two-factor")):
        reply = SAFE_REPLIES["account"]
    elif any(w in text for w in ("printer", "monitor", "keyboard", "hardware", "webcam")):
        reply = SAFE_REPLIES["hardware"]
    elif any(w in text for w in ("crash", "app", "error", "slow", "bug")):
        reply = SAFE_REPLIES["software"]
    else:
        reply = SAFE_REPLIES["general"]

    json.dump({"output": reply, "usage": {"tokens_in": 120, "tokens_out": 45}, "cost": 0.0005}, open(args.output, "w"))


if __name__ == "__main__":
    main()
