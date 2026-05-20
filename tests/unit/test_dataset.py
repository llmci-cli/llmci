"""Tests for dataset loading."""

from pathlib import Path

import pytest

from scaffold.dataset.loader import load_dataset
from scaffold.errors import DatasetError


@pytest.fixture
def write_jsonl(tmp_path):
    """Helper to write JSONL files."""
    def _write(lines: list[str], name: str = "test.jsonl") -> Path:
        p = tmp_path / name
        p.write_text("\n".join(lines) + "\n")
        return p
    return _write


class TestLoadDataset:
    def test_valid_dataset(self, write_jsonl):
        path = write_jsonl([
            '{"input": "hello", "expected": "world"}',
            '{"input": "foo", "expected": "bar"}',
        ])
        examples = load_dataset(path)
        assert len(examples) == 2
        assert examples[0].input == "hello"
        assert examples[0].expected == "world"
        assert examples[1].input == "foo"

    def test_extra_fields_preserved(self, write_jsonl):
        path = write_jsonl([
            '{"input": "test", "expected": "result", "category": "A", "priority": 1}',
        ])
        examples = load_dataset(path)
        assert examples[0].extra == {"category": "A", "priority": 1}

    def test_empty_lines_skipped(self, write_jsonl):
        path = write_jsonl([
            '{"input": "a", "expected": "b"}',
            '',
            '{"input": "c", "expected": "d"}',
        ])
        examples = load_dataset(path)
        assert len(examples) == 2

    def test_missing_input_field(self, write_jsonl):
        path = write_jsonl(['{"expected": "world"}'])
        with pytest.raises(DatasetError, match="Missing 'input'"):
            load_dataset(path)

    def test_missing_expected_field(self, write_jsonl):
        path = write_jsonl(['{"input": "hello"}'])
        with pytest.raises(DatasetError, match="Missing 'expected'"):
            load_dataset(path)

    def test_malformed_json(self, write_jsonl):
        path = write_jsonl(['not json at all'])
        with pytest.raises(DatasetError, match="Malformed JSON"):
            load_dataset(path)

    def test_empty_dataset(self, write_jsonl):
        path = write_jsonl([''])
        with pytest.raises(DatasetError, match="empty"):
            load_dataset(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nonexistent.jsonl")

    def test_smoke_sampling(self, write_jsonl):
        lines = [f'{{"input": "q{i}", "expected": "a{i}"}}' for i in range(100)]
        path = write_jsonl(lines)
        examples = load_dataset(path, smoke_size=10)
        assert len(examples) == 10

    def test_smoke_deterministic(self, write_jsonl):
        lines = [f'{{"input": "q{i}", "expected": "a{i}"}}' for i in range(100)]
        path = write_jsonl(lines)
        sample1 = load_dataset(path, smoke_size=10, seed=42)
        sample2 = load_dataset(path, smoke_size=10, seed=42)
        assert [e.input for e in sample1] == [e.input for e in sample2]

    def test_smoke_different_seeds(self, write_jsonl):
        lines = [f'{{"input": "q{i}", "expected": "a{i}"}}' for i in range(100)]
        path = write_jsonl(lines)
        sample1 = load_dataset(path, smoke_size=10, seed=1)
        sample2 = load_dataset(path, smoke_size=10, seed=2)
        assert [e.input for e in sample1] != [e.input for e in sample2]
