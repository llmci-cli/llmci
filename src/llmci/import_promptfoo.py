"""Convert Promptfoo configs to llmci.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from llmci.errors import ConfigError


def import_promptfoo_config(
    source: Path, output: Path = Path("llmci.yaml")
) -> None:
    """Parse a promptfooconfig.yaml and convert to llmci.yaml."""
    if not source.exists():
        raise ConfigError(f"Source file not found: {source}")

    try:
        raw = yaml.safe_load(source.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {source}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Expected a YAML mapping in {source}")

    config: dict = {"version": 1}
    warnings: list[str] = []

    config["target"] = _convert_providers(raw, warnings)

    evals, dataset_path = _convert_tests(raw, warnings)
    config["evals"] = evals

    config["settings"] = {"parallelism": 5, "timeout_per_call": 30, "retries": 1}

    if raw.get("prompts"):
        prompts = raw["prompts"]
        if isinstance(prompts, list) and len(prompts) == 1:
            prompt_file = prompts[0]
            if isinstance(prompt_file, str) and not prompt_file.startswith("file://"):
                (output.parent / "prompt.txt").write_text(prompt_file)
                config["target"]["prompt_file"] = "prompt.txt"
            elif isinstance(prompt_file, str):
                config["target"]["prompt_file"] = prompt_file.replace("file://", "")
        elif isinstance(prompts, list) and len(prompts) > 1:
            warnings.append(
                "Multiple prompts found — only the first is used. "
                "llmci uses a single prompt file."
            )
            prompt_file = prompts[0]
            if isinstance(prompt_file, str):
                (output.parent / "prompt.txt").write_text(prompt_file)
                config["target"]["prompt_file"] = "prompt.txt"

    yaml_text = yaml.dump(config, default_flow_style=False, sort_keys=False)
    output.write_text(yaml_text)

    if dataset_path:
        # Write the dataset next to the generated config, not relative to cwd, while the
        # stored config path stays relative so the generated project remains portable.
        _write_dataset(raw, output.parent / dataset_path)

    for w in warnings:
        print(f"  Warning: {w}")


def _convert_providers(raw: dict, warnings: list[str]) -> dict:
    """Convert promptfoo providers to llmci target config."""
    providers = raw.get("providers", [])
    if not providers:
        return {"command": "echo {input_file}"}

    provider = providers[0] if isinstance(providers, list) else providers

    if isinstance(provider, str):
        parts = provider.split(":")
        if len(parts) >= 2:
            return {
                "direct": {"provider": parts[0], "model": ":".join(parts[1:])}
            }
        return {"direct": {"provider": "openai", "model": provider}}

    if isinstance(provider, dict):
        provider_id = provider.get("id", "")
        parts = provider_id.split(":")
        if len(parts) >= 2:
            return {
                "direct": {"provider": parts[0], "model": ":".join(parts[1:])}
            }

        if "config" in provider:
            warnings.append(
                f"Provider config options ignored: {list(provider['config'].keys())}"
            )

    warnings.append("Could not parse provider — using command mode placeholder.")
    return {"command": "echo {input_file}"}


def _convert_tests(raw: dict, warnings: list[str]) -> tuple[list[dict], str | None]:
    """Convert promptfoo tests to llmci evals."""
    tests = raw.get("tests", [])
    if not tests:
        return [
            {
                "name": "default",
                "dataset": "./evals/default.jsonl",
                "judge": "exact_match",
                "metrics": [
                    {"name": "accuracy", "threshold": 0.90, "mode": "absolute"}
                ],
            }
        ], None

    has_assert = any(
        "assert" in t for t in tests if isinstance(t, dict)
    )

    judge: str | dict = "exact_match"
    metrics = [{"name": "accuracy", "threshold": 0.90, "mode": "absolute"}]

    if has_assert:
        assert_types = set()
        for t in tests:
            if isinstance(t, dict) and "assert" in t:
                for a in t["assert"]:
                    if isinstance(a, dict):
                        assert_types.add(a.get("type", ""))

        if "llm-rubric" in assert_types or "model-graded-closedqa" in assert_types:
            judge = {
                "type": "llm",
                "model": "gpt-4o-mini",
                "rubric": "Evaluate for accuracy and relevance.",
            }
            metrics = [
                {"name": "mean_score", "threshold": 0.70, "mode": "absolute"}
            ]

        unsupported = assert_types - {
            "equals", "contains", "is-json", "llm-rubric",
            "model-graded-closedqa", "similar",
        }
        if unsupported:
            warnings.append(
                f"Unsupported assert types ignored: {', '.join(sorted(unsupported))}"
            )

    dataset_path = "./evals/imported.jsonl"

    eval_cfg = {
        "name": "imported",
        "dataset": dataset_path,
        "judge": judge,
        "metrics": metrics,
    }

    if raw.get("redTeam"):
        warnings.append("Red teaming plugins are not supported by llmci.")

    return [eval_cfg], dataset_path


def _write_dataset(raw: dict, dataset_path: str | Path) -> None:
    """Extract test vars into a JSONL dataset."""
    tests = raw.get("tests", [])
    if not tests:
        return

    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for test in tests:
            if not isinstance(test, dict):
                continue

            vars_data = test.get("vars", {})
            if not isinstance(vars_data, dict):
                continue

            input_val = vars_data.get("input", vars_data.get("query", ""))
            expected_val = ""

            asserts = test.get("assert", [])
            for a in asserts:
                if isinstance(a, dict) and a.get("value"):
                    expected_val = str(a["value"])
                    break

            if not expected_val:
                expected_val = str(vars_data.get("expected", vars_data.get("output", "")))

            if input_val:
                row = {"input": str(input_val), "expected": str(expected_val)}
                f.write(json.dumps(row) + "\n")
