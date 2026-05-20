#!/usr/bin/env python3
"""Mock API that returns JSON responses.

Usage: python3 run_api.py --input {input_file} --output {output_file}
"""

import json


def handle_request(input_text: str) -> str:
    """Return a JSON response based on the input."""
    if "user_id=123" in input_text and "orders" not in input_text.lower():
        return json.dumps({"user_id": 123, "name": "Alice", "email": "alice@example.com"})
    elif "user_id=456" in input_text:
        return json.dumps({"user_id": 456, "name": "Bob", "email": "bob@example.com"})
    elif "customer_id=123" in input_text:
        return json.dumps({"customer_id": 123, "orders": [{"id": 1, "total": 29.99}]})
    elif "sku=ABC-100" in input_text:
        return json.dumps({"sku": "ABC-100", "name": "Widget", "price": 9.99, "in_stock": True})
    elif "name=Charlie" in input_text:
        return json.dumps({"results": [{"user_id": 789, "name": "Charlie"}], "count": 1})
    else:
        return json.dumps({"error": "Unknown request"})


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    result = handle_request(data["input"])
    json.dump({"output": result}, open(args.output, "w"))


if __name__ == "__main__":
    main()
