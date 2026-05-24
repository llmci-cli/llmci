#!/usr/bin/env python3
"""llmci eval entrypoint for OpenAI Agents SDK integration.

MOCK_LLM=1  — deterministic mock using TraceBuilder (CI, no API key)
default     — OpenAI Agents SDK (requires OPENAI_API_KEY and pip install llmci[agents])

Usage: python3 run_agent.py --input {input_file} --output {output_file}
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def run(input_data: dict) -> dict:
    if os.environ.get("MOCK_LLM", "0") == "1":
        from mock_agent import run_mock

        return run_mock(input_data)

    try:
        from agent_def import build_agent
        from llmci.integrations.openai_agents import run_for_llmci_sync
    except ImportError as e:
        raise SystemExit(
            "OpenAI Agents path requires: pip install 'llmci[agents]' and OPENAI_API_KEY. "
            "For CI/mock runs use MOCK_LLM=1."
        ) from e

    return run_for_llmci_sync(build_agent(), input_data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_data = json.loads(open(args.input).read())
    result = run(input_data)
    json.dump(result, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
