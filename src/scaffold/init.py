"""Interactive scaffold.yaml generator."""

from __future__ import annotations

import json
from pathlib import Path

import click
import yaml

STARTER_EXAMPLES: dict[str, list[dict]] = {
    "classification": [
        {"input": "My printer won't connect to wifi", "expected": "hardware"},
        {"input": "I need a refund for my last order", "expected": "billing"},
        {"input": "How do I reset my password?", "expected": "account"},
        {"input": "The app crashes on startup", "expected": "software"},
        {"input": "Can I upgrade my subscription?", "expected": "billing"},
    ],
    "open_ended": [
        {
            "input": "Summarize the benefits of renewable energy",
            "expected": "A concise summary covering solar, wind, and hydro benefits",
        },
        {
            "input": "Explain quantum computing to a 10 year old",
            "expected": "A simple analogy-based explanation without jargon",
        },
        {
            "input": "Write a professional email declining a meeting",
            "expected": "A polite, concise email with a reason and alternative",
        },
    ],
    "agent": [
        {
            "input": {"query": "What's the weather in New York?"},
            "expected": {
                "outcome": "weather information for New York",
                "constraints": {"max_tool_calls": 3},
            },
        },
        {
            "input": {"query": "Look up order #1234"},
            "expected": {
                "outcome": "order details for #1234",
                "constraints": {"max_tool_calls": 5},
            },
        },
    ],
}


def run_init() -> None:
    """Run the interactive scaffold init flow."""
    if Path("scaffold.yaml").exists():
        if not click.confirm("scaffold.yaml already exists. Overwrite?", default=False):
            click.echo("Aborted.")
            return

    prompt_files = _detect_prompt_files()
    if prompt_files:
        click.echo(f"Detected prompt file(s): {', '.join(prompt_files)}")

    mode = click.prompt(
        "\nTarget mode",
        type=click.Choice(["command", "direct"]),
        default="command",
    )

    task_type = click.prompt(
        "Task type",
        type=click.Choice(["classification", "open_ended", "agent"]),
        default="classification",
    )

    eval_name = click.prompt("Eval name", default="my-eval")

    config: dict = {"version": 1}

    if mode == "command":
        default_cmd = "python3 run.py --input {input_file} --output {output_file}"
        command = click.prompt("Command template", default=default_cmd)
        config["target"] = {"command": command}
    else:
        provider = click.prompt("Provider", default="openai")
        model = click.prompt("Model", default="gpt-4o-mini")
        config["target"] = {"direct": {"provider": provider, "model": model}}
        if prompt_files:
            config["target"]["prompt_file"] = prompt_files[0]

    dataset_path = f"./evals/{eval_name}.jsonl"

    eval_cfg: dict = {
        "name": eval_name,
        "dataset": dataset_path,
    }

    if task_type == "classification":
        eval_cfg["judge"] = "exact_match"
        eval_cfg["metrics"] = [
            {"name": "accuracy", "threshold": 0.90, "mode": "absolute"},
            {"name": "f1_macro", "threshold": 0.85, "mode": "absolute"},
        ]
    elif task_type == "open_ended":
        judge_model = click.prompt("Judge model", default="gpt-4o-mini")
        eval_cfg["judge"] = {
            "type": "llm",
            "model": judge_model,
            "rubric": "Evaluate the response for accuracy, completeness, and clarity.",
        }
        eval_cfg["metrics"] = [
            {"name": "mean_score", "threshold": 0.70, "mode": "absolute"},
        ]
    else:
        eval_cfg["level"] = "agent"
        eval_cfg["judge"] = {
            "type": "composite",
            "criteria": [
                {"name": "constraints", "type": "constraint", "weight": 1.0},
            ],
        }
        eval_cfg["metrics"] = [
            {"name": "mean_score", "threshold": 0.70, "mode": "absolute"},
        ]

    config["evals"] = [eval_cfg]
    config["settings"] = {
        "parallelism": 5,
        "timeout_per_call": 30,
        "retries": 1,
    }

    yaml_text = yaml.dump(config, default_flow_style=False, sort_keys=False)
    Path("scaffold.yaml").write_text(yaml_text)
    click.echo("\nCreated scaffold.yaml")

    evals_dir = Path("evals")
    evals_dir.mkdir(exist_ok=True)
    dataset_file = evals_dir / f"{eval_name}.jsonl"

    examples: list[dict] = STARTER_EXAMPLES.get(
        task_type, STARTER_EXAMPLES["classification"]
    )
    with dataset_file.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    click.echo(f"Created {dataset_file} with {len(examples)} starter examples")

    click.echo("\nNext steps:")
    click.echo(f"  1. Edit {dataset_file} — add your real eval examples")
    if mode == "command":
        click.echo("  2. Create your target script (reads --input JSON, writes --output JSON)")
    else:
        click.echo("  2. Set your API key (e.g. export OPENAI_API_KEY=...)")
    click.echo("  3. Run: scaffold run")


def _detect_prompt_files() -> list[str]:
    """Look for common prompt file patterns in the current directory."""
    patterns = ["prompt.txt", "prompt.md", "system_prompt.txt", "*.prompt"]
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(p) for p in Path(".").glob(pattern))
    return sorted(set(found))
