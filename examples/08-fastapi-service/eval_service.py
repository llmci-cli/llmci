#!/usr/bin/env python3
"""Service-level eval wrapper.

Tests the FULL pipeline: preprocess → classify → postprocess.
This catches regressions from PII redaction, confidence thresholds,
or any other non-prompt changes that affect end-to-end predictions.

Usage: python3 eval_service.py --input {input_file} --output {output_file}
"""

import argparse
import json

from service import classify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    result = classify(data["input"])
    json.dump({"output": result["category"]}, open(args.output, "w"))


if __name__ == "__main__":
    main()
