"""Import datasets from CSV or JSON into JSONL format."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scaffold.errors import DatasetError


def import_dataset(
    name: str,
    source_path: str | Path,
    base_dir: Path = Path("evals"),
    input_column: str = "input",
    expected_column: str = "expected",
) -> tuple[int, int]:
    """Import from CSV or JSON into JSONL format.

    Returns (imported_count, skipped_count).
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise DatasetError(f"Source file not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(source_path, input_column, expected_column)
    elif suffix == ".json":
        rows = _read_json(source_path, input_column, expected_column)
    else:
        raise DatasetError(
            f"Unsupported file format: {suffix}. Use .csv or .json"
        )

    base_dir.mkdir(exist_ok=True)
    output_path = base_dir / f"{name}.jsonl"

    imported = 0
    skipped = 0

    with output_path.open("a") as f:
        for row in rows:
            if row is None:
                skipped += 1
                continue
            f.write(json.dumps(row) + "\n")
            imported += 1

    return imported, skipped


def _read_csv(
    path: Path, input_col: str, expected_col: str
) -> list[dict | None]:
    """Read CSV and extract input/expected columns."""
    rows: list[dict | None] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise DatasetError(f"CSV file has no header: {path}")

        if input_col not in reader.fieldnames:
            raise DatasetError(
                f"CSV missing '{input_col}' column. "
                f"Available: {', '.join(reader.fieldnames)}"
            )
        if expected_col not in reader.fieldnames:
            raise DatasetError(
                f"CSV missing '{expected_col}' column. "
                f"Available: {', '.join(reader.fieldnames)}"
            )

        for row_num, row in enumerate(reader, start=2):
            input_val = row.get(input_col, "").strip()
            expected_val = row.get(expected_col, "").strip()

            if not input_val or not expected_val:
                rows.append(None)
                continue

            entry: dict = {"input": input_val, "expected": expected_val}
            for k, v in row.items():
                if k not in (input_col, expected_col) and v:
                    entry[k] = v
            rows.append(entry)

    return rows


def _read_json(
    path: Path, input_col: str, expected_col: str
) -> list[dict | None]:
    """Read JSON array and extract input/expected fields."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise DatasetError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, list):
        raise DatasetError(
            f"Expected JSON array in {path}, got {type(data).__name__}"
        )

    rows: list[dict | None] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            rows.append(None)
            continue

        input_val = str(item.get(input_col, "")).strip()
        expected_val = str(item.get(expected_col, "")).strip()

        if not input_val or not expected_val:
            rows.append(None)
            continue

        entry: dict = {"input": input_val, "expected": expected_val}
        for k, v in item.items():
            if k not in (input_col, expected_col):
                entry[k] = v
        rows.append(entry)

    return rows
