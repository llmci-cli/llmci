#!/usr/bin/env python3
"""Prompt-level eval wrapper.

Tests ONLY the classification logic in isolation — no pre/post processing.
This catches prompt regressions quickly without service startup.

Usage: python3 run_prompt.py --input {input_file} --output {output_file}
"""

import argparse
import json

from service import classify_core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    category, _ = classify_core(data["input"])
    json.dump({"output": category}, open(args.output, "w"))


if __name__ == "__main__":
    main()
