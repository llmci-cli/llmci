"""Tests for dataset coverage analysis."""

import json

import pytest

from scaffold.dataset.check import check_dataset, format_coverage_report


@pytest.fixture
def write_jsonl(tmp_path):
    def _write(rows: list[dict], name: str = "test.jsonl"):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p
    return _write


class TestCheckDataset:
    def test_healthy_dataset(self, write_jsonl):
        rows = [
            {"input": f"q{i}", "expected": cat}
            for i, cat in enumerate(["a"] * 10 + ["b"] * 10)
        ]
        path = write_jsonl(rows)
        report = check_dataset(path, min_per_category=5)
        assert report.total_examples == 20
        assert report.categories == {"a": 10, "b": 10}
        assert not report.warnings

    def test_small_dataset_warning(self, write_jsonl):
        rows = [{"input": f"q{i}", "expected": "a"} for i in range(5)]
        path = write_jsonl(rows)
        report = check_dataset(path)
        assert any("small" in w.lower() for w in report.warnings)

    def test_underrepresented_category(self, write_jsonl):
        rows = (
            [{"input": f"q{i}", "expected": "a"} for i in range(20)]
            + [{"input": f"r{i}", "expected": "b"} for i in range(2)]
        )
        path = write_jsonl(rows)
        report = check_dataset(path, min_per_category=5)
        assert any("'b'" in w for w in report.warnings)
        assert any("'b'" in s for s in report.suggestions)

    def test_duplicates_detected(self, write_jsonl):
        rows = [
            {"input": "same question", "expected": "a"},
            {"input": "same question", "expected": "a"},
            {"input": "different", "expected": "b"},
        ]
        path = write_jsonl(rows)
        report = check_dataset(path)
        assert len(report.duplicates) == 1
        assert "same question" in report.duplicates

    def test_single_category_warning(self, write_jsonl):
        rows = [{"input": f"q{i}", "expected": "only"} for i in range(20)]
        path = write_jsonl(rows)
        report = check_dataset(path)
        assert any("same expected" in w.lower() for w in report.warnings)

    def test_class_imbalance_warning(self, write_jsonl):
        rows = (
            [{"input": f"q{i}", "expected": "a"} for i in range(50)]
            + [{"input": f"r{i}", "expected": "b"} for i in range(5)]
        )
        path = write_jsonl(rows)
        report = check_dataset(path)
        assert any("imbalance" in w.lower() for w in report.warnings)

    def test_length_stats(self, write_jsonl):
        rows = [
            {"input": "short", "expected": "a"},
            {"input": "a medium length input", "expected": "a"},
            {"input": "this is a longer input string", "expected": "b"},
        ]
        path = write_jsonl(rows)
        report = check_dataset(path)
        assert "mean" in report.length_stats
        assert "std" in report.length_stats

    def test_format_report(self, write_jsonl):
        rows = [{"input": f"q{i}", "expected": "a"} for i in range(10)]
        path = write_jsonl(rows)
        report = check_dataset(path)
        text = format_coverage_report(report)
        assert "10 examples" in text
        assert "a" in text
