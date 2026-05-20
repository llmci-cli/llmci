"""Tests for dataset import from CSV/JSON."""

import csv
import json

import pytest

from scaffold.dataset.import_data import import_dataset
from scaffold.errors import DatasetError


@pytest.fixture
def evals_dir(tmp_path):
    d = tmp_path / "evals"
    d.mkdir()
    return d


class TestImportCSV:
    def test_basic_csv(self, tmp_path, evals_dir):
        csv_path = tmp_path / "data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "expected"])
            writer.writerow(["hello", "world"])
            writer.writerow(["foo", "bar"])

        imported, skipped = import_dataset("test", csv_path, base_dir=evals_dir)
        assert imported == 2
        assert skipped == 0

        output = (evals_dir / "test.jsonl").read_text().strip().split("\n")
        assert len(output) == 2
        row = json.loads(output[0])
        assert row["input"] == "hello"
        assert row["expected"] == "world"

    def test_csv_with_extra_columns(self, tmp_path, evals_dir):
        csv_path = tmp_path / "data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "expected", "category"])
            writer.writerow(["q1", "a1", "cat_a"])

        imported, _ = import_dataset("test", csv_path, base_dir=evals_dir)
        assert imported == 1
        row = json.loads((evals_dir / "test.jsonl").read_text().strip())
        assert row["category"] == "cat_a"

    def test_csv_missing_column(self, tmp_path, evals_dir):
        csv_path = tmp_path / "data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            writer.writerow(["q1", "a1"])

        with pytest.raises(DatasetError, match="input"):
            import_dataset("test", csv_path, base_dir=evals_dir)

    def test_csv_custom_columns(self, tmp_path, evals_dir):
        csv_path = tmp_path / "data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            writer.writerow(["q1", "a1"])

        imported, _ = import_dataset(
            "test", csv_path, base_dir=evals_dir,
            input_column="question", expected_column="answer",
        )
        assert imported == 1

    def test_csv_skips_empty_rows(self, tmp_path, evals_dir):
        csv_path = tmp_path / "data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "expected"])
            writer.writerow(["q1", "a1"])
            writer.writerow(["", "a2"])
            writer.writerow(["q3", ""])

        imported, skipped = import_dataset("test", csv_path, base_dir=evals_dir)
        assert imported == 1
        assert skipped == 2


class TestImportJSON:
    def test_basic_json(self, tmp_path, evals_dir):
        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps([
            {"input": "hello", "expected": "world"},
            {"input": "foo", "expected": "bar"},
        ]))

        imported, skipped = import_dataset("test", json_path, base_dir=evals_dir)
        assert imported == 2
        assert skipped == 0

    def test_json_not_array(self, tmp_path, evals_dir):
        json_path = tmp_path / "data.json"
        json_path.write_text('{"input": "hello"}')

        with pytest.raises(DatasetError, match="array"):
            import_dataset("test", json_path, base_dir=evals_dir)

    def test_json_skips_invalid_items(self, tmp_path, evals_dir):
        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps([
            {"input": "q1", "expected": "a1"},
            "not a dict",
            {"input": "q2"},
        ]))

        imported, skipped = import_dataset("test", json_path, base_dir=evals_dir)
        assert imported == 1
        assert skipped == 2

    def test_unsupported_format(self, tmp_path, evals_dir):
        txt_path = tmp_path / "data.txt"
        txt_path.write_text("hello")

        with pytest.raises(DatasetError, match="Unsupported"):
            import_dataset("test", txt_path, base_dir=evals_dir)

    def test_missing_source(self, evals_dir):
        with pytest.raises(DatasetError, match="not found"):
            import_dataset("test", "/nonexistent/data.csv", base_dir=evals_dir)
