"""JSONL dataset loading and validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from scaffold.errors import DatasetError
from scaffold.models import EvalExample


def load_dataset(
    path: str | Path,
    smoke_size: int | None = None,
    seed: int = 42,
) -> list[EvalExample]:
    """Load a JSONL eval dataset.

    Each line must be a JSON object with at minimum 'input' and 'expected' fields.
    Additional fields are preserved in the 'extra' dict.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found: {path}\n\n"
            "Fix: Create the dataset file, or check the path in scaffold.yaml."
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
                raise DatasetError(
                    f"Missing 'expected' field at {path} line {line_num}:\n"
                    f"  {line[:200]}\n\n"
                    "Fix: Each line must have an 'expected' field."
                )

            extra = {k: v for k, v in row.items() if k not in ("input", "expected")}
            examples.append(
                EvalExample(input=row["input"], expected=row["expected"], extra=extra)
            )

    if not examples:
        raise DatasetError(f"Dataset is empty: {path}")

    if smoke_size is not None and smoke_size < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, smoke_size)

    return examples
