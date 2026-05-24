"""JSONL dataset loading and validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from llmci.errors import DatasetError
from llmci.models import AgentScenario, EvalExample


def load_dataset(
    path: str | Path,
    smoke_size: int | None = None,
    seed: int = 42,
    require_expected: bool = True,
) -> list[EvalExample]:
    """Load a JSONL eval dataset.

    Each line must be a JSON object with at minimum an 'input' field.
    The 'expected' field is required by default but can be made optional
    for LLM-as-judge evals that only need input + output.
    Additional fields are preserved in the 'extra' dict.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found: {path}\n\n"
            "Fix: Create the dataset file, or check the path in llmci.yaml."
        )

    examples: list[EvalExample] = []

    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(
                    f"Malformed JSON at {path} line {line_num}:\n"
                    f"  {line[:200]}\n"
                    f"  Error: {e}\n\n"
                    "Fix: Each line must be a valid JSON object."
                ) from e

            if not isinstance(row, dict):
                raise DatasetError(
                    f"Expected JSON object at {path} line {line_num}, "
                    f"got {type(row).__name__}"
                )

            if "input" not in row:
                raise DatasetError(
                    f"Missing 'input' field at {path} line {line_num}:\n"
                    f"  {line[:200]}\n\n"
                    "Fix: Each line must have an 'input' field."
                )

            if "expected" not in row:
                if require_expected:
                    raise DatasetError(
                        f"Missing 'expected' field at {path} line {line_num}:\n"
                        f"  {line[:200]}\n\n"
                        "Fix: Each line must have an 'expected' field "
                        "(or use an LLM judge which doesn't require one)."
                    )

            extra = {k: v for k, v in row.items() if k not in ("input", "expected")}
            examples.append(
                EvalExample(
                    input=row["input"],
                    expected=row.get("expected", ""),
                    extra=extra,
                )
            )

    if not examples:
        raise DatasetError(f"Dataset is empty: {path}")

    if smoke_size is not None and smoke_size < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, smoke_size)

    return examples


def load_agent_scenarios(
    path: str | Path,
    smoke_size: int | None = None,
    seed: int = 42,
) -> list[AgentScenario]:
    """Load agent scenarios from a JSONL file.

    Each line must be a JSON object with either:
    - (input + expected) for single-turn scenarios
    - (turns) for multi-turn scenarios
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Agent dataset not found: {path}")

    scenarios: list[AgentScenario] = []

    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(
                    f"Malformed JSON at {path} line {line_num}: {e}"
                ) from e

            if not isinstance(row, dict):
                raise DatasetError(
                    f"Expected JSON object at {path} line {line_num}"
                )

            try:
                scenarios.append(AgentScenario(**row))
            except Exception as e:
                raise DatasetError(
                    f"Invalid agent scenario at {path} line {line_num}: {e}"
                ) from e

    if not scenarios:
        raise DatasetError(f"Agent dataset is empty: {path}")

    if smoke_size is not None and smoke_size < len(scenarios):
        rng = random.Random(seed)
        scenarios = rng.sample(scenarios, smoke_size)

    return scenarios
