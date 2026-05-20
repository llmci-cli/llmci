"""Tests for baseline storage."""

import json
from unittest.mock import patch

import pytest

from scaffold.baseline import (
    load_all_baselines,
    load_baseline,
    save_baseline,
)
from scaffold.models import EvalResult


@pytest.fixture
def mock_baseline_dir(tmp_path, monkeypatch):
    """Point BASELINE_DIR to a temp directory."""
    bl_dir = tmp_path / ".scaffold" / "baselines"
    monkeypatch.setattr("scaffold.baseline.BASELINE_DIR", bl_dir)
    return bl_dir


@pytest.fixture
def sample_result():
    return EvalResult(
        eval_name="test-eval",
        metrics={"accuracy": 0.95, "f1_macro": 0.92},
        num_examples=20,
    )


class TestSaveBaseline:
    def test_creates_file(self, mock_baseline_dir, sample_result):
        path = save_baseline(sample_result, commit_sha="abc123")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["eval_name"] == "test-eval"
        assert data["metrics"]["accuracy"] == 0.95
        assert data["commit_sha"] == "abc123"

    def test_creates_directory(self, mock_baseline_dir, sample_result):
        assert not mock_baseline_dir.exists()
        save_baseline(sample_result, commit_sha="abc123")
        assert mock_baseline_dir.exists()

    def test_overwrites_existing(self, mock_baseline_dir, sample_result):
        save_baseline(sample_result, commit_sha="first")
        sample_result.metrics["accuracy"] = 0.98
        save_baseline(sample_result, commit_sha="second")
        data = json.loads((mock_baseline_dir / "test-eval.json").read_text())
        assert data["metrics"]["accuracy"] == 0.98
        assert data["commit_sha"] == "second"


class TestLoadBaseline:
    def test_load_from_disk(self, mock_baseline_dir, sample_result):
        save_baseline(sample_result, commit_sha="abc123")
        bl = load_baseline("test-eval")
        assert bl is not None
        assert bl.eval_name == "test-eval"
        assert bl.metrics["accuracy"] == 0.95

    def test_missing_baseline(self, mock_baseline_dir):
        bl = load_baseline("nonexistent")
        assert bl is None

    def test_corrupt_baseline(self, mock_baseline_dir):
        mock_baseline_dir.mkdir(parents=True)
        (mock_baseline_dir / "corrupt.json").write_text("not json")
        bl = load_baseline("corrupt")
        assert bl is None

    def test_load_from_git_ref(self, mock_baseline_dir):
        baseline_json = json.dumps({
            "eval_name": "test-eval",
            "metrics": {"accuracy": 0.90},
            "timestamp": "2025-01-01T00:00:00",
            "commit_sha": "def456",
        })

        with patch("scaffold.baseline.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = baseline_json
            bl = load_baseline("test-eval", ref="origin/main")

        assert bl is not None
        assert bl.metrics["accuracy"] == 0.90
        assert bl.commit_sha == "def456"

    def test_load_from_git_ref_not_found(self, mock_baseline_dir):
        with patch("scaffold.baseline.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 128
            bl = load_baseline("test-eval", ref="origin/main")

        assert bl is None


class TestLoadAllBaselines:
    def test_loads_multiple(self, mock_baseline_dir):
        for name, acc in [("eval1", 0.9), ("eval2", 0.8)]:
            result = EvalResult(
                eval_name=name,
                metrics={"accuracy": acc},
                num_examples=10,
            )
            save_baseline(result, commit_sha="abc")

        baselines = load_all_baselines(["eval1", "eval2"])
        assert len(baselines) == 2
        assert baselines["eval1"].metrics["accuracy"] == 0.9
        assert baselines["eval2"].metrics["accuracy"] == 0.8

    def test_missing_baselines_omitted(self, mock_baseline_dir):
        result = EvalResult(
            eval_name="eval1",
            metrics={"accuracy": 0.9},
            num_examples=10,
        )
        save_baseline(result, commit_sha="abc")

        baselines = load_all_baselines(["eval1", "eval2"])
        assert len(baselines) == 1
        assert "eval1" in baselines
        assert "eval2" not in baselines
