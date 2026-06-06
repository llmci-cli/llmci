"""Tests for judge calibration and drift detection."""

import asyncio

import pytest

from llmci.calibrate import (
    CalibrationResult,
    DriftResult,
    LabeledExample,
    _cohens_kappa,
    _pearson,
    compute_agreement,
    compute_drift,
    format_calibration_report,
    load_labeled_set,
    load_snapshot,
    run_calibration,
    save_snapshot,
)
from llmci.errors import DatasetError
from llmci.models import JudgeResult


class _ScriptedJudge:
    """Judge that returns a preset score per example index (by output)."""

    def __init__(self, scores):
        self._scores = scores

    async def evaluate_dataset(self, examples, results):
        return [JudgeResult(score=self._scores[i]) for i in range(len(results))]


class TestLoadLabeledSet:
    def test_loads_and_normalizes(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"input": "a", "output": "x", "human_score": 1}\n'
            '{"input": "b", "output": "y", "human_score": "fail"}\n'
            '{"input": "c", "output": "z", "human_score": 0.5, "expected": "e"}\n'
        )
        labeled = load_labeled_set(path)
        assert [le.human_score for le in labeled] == [1.0, 0.0, 0.5]
        assert labeled[2].expected == "e"

    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError):
            load_labeled_set(tmp_path / "nope.jsonl")

    def test_missing_fields(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"input": "a", "output": "x"}\n')
        with pytest.raises(DatasetError):
            load_labeled_set(path)

    def test_empty(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n")
        with pytest.raises(DatasetError):
            load_labeled_set(path)


class TestAgreement:
    def test_perfect_agreement(self):
        agreement, kappa, mae, pearson = compute_agreement(
            [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]
        )
        assert agreement == 1.0
        assert kappa == 1.0
        assert mae == 0.0
        assert pearson == pytest.approx(1.0)

    def test_total_disagreement(self):
        agreement, kappa, _, _ = compute_agreement([1.0, 1.0], [0.0, 0.0])
        assert agreement == 0.0
        assert kappa <= 0.0

    def test_empty(self):
        assert compute_agreement([], []) == (0.0, 0.0, 0.0, 0.0)

    def test_kappa_chance(self):
        # All judge=pass, humans split → agreement by chance, kappa ~0
        kappa = _cohens_kappa([True, True, True, True], [True, True, False, False])
        assert kappa == pytest.approx(0.0)

    def test_pearson_zero_variance(self):
        assert _pearson([0.5, 0.5, 0.5], [1.0, 0.0, 1.0]) == 0.0


class TestRunCalibration:
    def test_runs_judge_and_scores(self):
        labeled = [
            LabeledExample("a", "", "out-a", 1.0),
            LabeledExample("b", "", "out-b", 0.0),
        ]
        judge = _ScriptedJudge([1.0, 0.0])
        result = asyncio.run(run_calibration(judge, "gpt-4o-mini", labeled))
        assert result.n == 2
        assert result.agreement_rate == 1.0
        assert result.judge_scores == [1.0, 0.0]
        assert result.human_scores == [1.0, 0.0]
        assert result.inputs == ["a", "b"]


class TestSnapshotAndDrift:
    def _result(self, model, scores, inputs):
        return CalibrationResult(
            model=model, n=len(scores), agreement_rate=1.0, cohens_kappa=1.0,
            mae=0.0, pearson=1.0, judge_scores=scores, human_scores=scores, inputs=inputs,
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        result = self._result("m1", [0.8, 0.2], ["a", "b"])
        save_snapshot("myeval", result, snapshot_dir=tmp_path)
        loaded = load_snapshot("myeval", snapshot_dir=tmp_path)
        assert loaded["model"] == "m1"
        assert loaded["scores_by_input"] == {"a": 0.8, "b": 0.2}

    def test_load_missing_returns_none(self, tmp_path):
        assert load_snapshot("nope", snapshot_dir=tmp_path) is None

    def test_load_corrupt_returns_none(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not json")
        assert load_snapshot("bad", snapshot_dir=tmp_path) is None

    def test_drift_no_snapshot(self):
        result = self._result("m1", [0.5], ["a"])
        assert compute_drift(result, None) is None

    def test_drift_model_changed(self):
        snapshot = {"model": "m1", "scores_by_input": {"a": 1.0, "b": 0.0}}
        result = self._result("m2", [0.5, 0.5], ["a", "b"])
        drift = compute_drift(result, snapshot)
        assert drift.model_changed is True
        assert drift.previous_model == "m1"
        assert drift.current_model == "m2"
        assert drift.mean_abs_change == pytest.approx(0.5)
        assert drift.n_compared == 2

    def test_drift_no_overlap(self):
        snapshot = {"model": "m1", "scores_by_input": {"z": 1.0}}
        result = self._result("m1", [0.5], ["a"])
        assert compute_drift(result, snapshot) is None


class TestReport:
    def test_report_contains_metrics(self):
        result = CalibrationResult(
            model="gpt-4o-mini", n=10, agreement_rate=0.9, cohens_kappa=0.75,
            mae=0.1, pearson=0.8,
        )
        report = format_calibration_report(result)
        assert "Judge Calibration" in report
        assert "0.900" in report
        assert "substantial" in report

    def test_report_includes_drift(self):
        result = CalibrationResult(
            model="m2", n=2, agreement_rate=1.0, cohens_kappa=1.0, mae=0.0, pearson=1.0,
        )
        drift = DriftResult(
            previous_model="m1", current_model="m2", model_changed=True,
            mean_abs_change=0.3, n_compared=2,
        )
        report = format_calibration_report(result, drift)
        assert "Drift vs snapshot" in report
        assert "changed" in report
        assert "0.300" in report
