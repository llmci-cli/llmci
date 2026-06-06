#!/usr/bin/env python3
"""A deterministic mock extractor for the structured-output example.

Usage: python3 run_extractor.py --input {input_file} --output {output_file}

It "extracts" a product record as JSON from each input. Every record conforms to the
JSON Schema in ``llmci.yaml`` (right types, required fields, no extras), so the
``structured`` judge scores 1.0.

To watch the gate fail, set ``BROKEN = True``: the extractor emits a record with a
string price and a missing ``id`` — the schema validator catches both.
"""

import argparse
import json

BROKEN = False

# Keyed on the input phrase; each value is a schema-valid product record.
RECORDS = {
    "Add the Aeron chair, $1395, sku 1001": {
        "id": 1001,
        "name": "Aeron chair",
        "price": 1395.0,
        "in_stock": True,
    },
    "Logitech MX mouse for 99.99, id 1002": {
        "id": 1002,
        "name": "Logitech MX mouse",
        "price": 99.99,
        "in_stock": True,
    },
    "Standing desk, sold out, $640, sku 1003": {
        "id": 1003,
        "name": "Standing desk",
        "price": 640.0,
        "in_stock": False,
    },
    "USB-C cable, 2 dollars, item 1004": {
        "id": 1004,
        "name": "USB-C cable",
        "price": 2.0,
        "in_stock": True,
    },
}

BROKEN_RECORD = {"name": "Mystery item", "price": "free", "in_stock": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    record = BROKEN_RECORD if BROKEN else RECORDS.get(data["input"], BROKEN_RECORD)

    # The target's answer *is* the JSON record (as a string), which the judge parses.
    with open(args.output, "w") as f:
        json.dump({"output": json.dumps(record)}, f)


if __name__ == "__main__":
    main()
