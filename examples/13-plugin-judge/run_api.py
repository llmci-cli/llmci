#!/usr/bin/env python3
"""A deterministic mock JSON API for the plugin-judge example.

Usage: python3 run_api.py --input {input_file} --output {output_file}
"""

import argparse
import json

RESPONSES = {
    "get product ABC-100": {"id": "ABC-100", "name": "Widget", "price": 9.99},
    "get product XYZ-200": {"id": "XYZ-200", "name": "Gadget", "price": 19.99},
    "get user 7": {"id": 7, "name": "Alice"},
    "get user 8": {"id": 8, "name": "Bob"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    payload = RESPONSES.get(data["input"], {"error": "not found"})

    with open(args.output, "w") as f:
        json.dump({"output": json.dumps(payload)}, f)


if __name__ == "__main__":
    main()
