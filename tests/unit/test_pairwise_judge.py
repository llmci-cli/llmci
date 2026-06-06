"""Tests for the pairwise / preference judge."""

from unittest.mock import patch

import pytest

from llmci.baseline import Baseline, BaselineExample
from llmci.judges.factory import create_judge
from llmci.judges.pairwise import PairwiseJudge, _parse_winner
from llmci.metrics import compute_metrics
from llmci.models import EvalExample, JudgeConfig, JudgeResult, TargetResult


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]


def _mock_llm(content):
    async def _m(*args, **kwargs):
        return _Resp(content)

    return _m


class TestParseWinner:
    def test_b_wins(self):
        score, reason = _parse_winner('{"winner": "B", "reasoning": "clearer"}')
        assert score == 1.0
        assert "current preferred" in reason

    def test_a_wins(self):
        score, _ = _parse_winner('{"winner": "A", "reasoning": "x"}')
        assert score == 0.0

    def test_tie(self):
        score, _ = _parse_winner('{"winner": "tie"}')
        assert score == 0.5

    def test_code_fence(self):
        score, _ = _parse_winner('```json\n{"winner": "B"}\n```')
        assert score == 1.0

    def test_unparseable_is_tie(self):
        score, reason = _parse_winner("garbage")
        assert score == 0.5
        assert "unparseable" in reason


def _baseline(pairs) -> Baseline:
    return Baseline(
        eval_name="svc", metrics={}, timestamp="t", commit_sha="x",
        examples=[BaselineExample(input=i, output=o, score=s) for i, o, s in pairs],
    )


async def test_win_against_baseline():
    judge = PairwiseJudge(model="gpt-4o-mini")
    judge.set_baseline(_baseline([("q1", "old answer", 1.0)]))
    examples = [EvalExample(input="q1", expected="")]
    results = [TargetResult(output="new better answer", latency_ms=1.0)]

    with patch("llmci.judges.pairwise.litellm.acompletion",
               side_effect=_mock_llm('{"winner": "B", "reasoning": "better"}')):
        per_example = await judge.evaluate_dataset(examples, results)

    assert per_example[0].score == 1.0
    assert per_example[0].sub_scores["win_rate"] == 1.0


async def test_new_example_without_baseline_is_neutral():
    judge = PairwiseJudge()
    judge.set_baseline(_baseline([("q1", "old", 1.0)]))
    examples = [EvalExample(input="q2-new", expected="")]
    results = [TargetResult(output="answer", latency_ms=1.0)]

    per_example = await judge.evaluate_dataset(examples, results)
    assert per_example[0].score == 0.5
    assert "no baseline" in per_example[0].reason


async def test_set_baseline_none_clears():
    judge = PairwiseJudge()
    judge.set_baseline(None)
    examples = [EvalExample(input="q", expected="")]
    results = [TargetResult(output="a", latency_ms=1.0)]

    per_example = await judge.evaluate_dataset(examples, results)
    assert per_example[0].score == 0.5  # nothing to compare


def test_win_rate_aggregates_as_metric():
    examples = [EvalExample(input=f"q{i}", expected="") for i in range(3)]
    results = [TargetResult(output="a", latency_ms=1.0) for _ in range(3)]
    per_example = [
        JudgeResult(score=1.0, sub_scores={"win_rate": 1.0}),
        JudgeResult(score=0.0, sub_scores={"win_rate": 0.0}),
        JudgeResult(score=0.5, sub_scores={"win_rate": 0.5}),
    ]
    metrics = compute_metrics(examples, results, per_example, ["win_rate"])
    assert metrics["win_rate"] == pytest.approx(0.5)


def test_factory_builds_pairwise_with_criterion():
    judge = create_judge(JudgeConfig(type="pairwise", rubric="Which is more concise?"))
    assert isinstance(judge, PairwiseJudge)
    assert judge.criterion == "Which is more concise?"
