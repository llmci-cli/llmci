"""Tests for the RAG judge and sub-score metrics."""

from unittest.mock import patch

import pytest

from llmci.judges.factory import create_judge
from llmci.judges.rag import RagJudge, _parse_score, _retrieval_precision, _retrieval_recall
from llmci.metrics import compute_metrics
from llmci.models import EvalExample, JudgeConfig, JudgeResult, TargetResult


class _MockMessage:
    def __init__(self, content):
        self.content = content


class _MockResponse:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _MockMessage(content)})()]


def _mock_llm(content):
    async def _m(*args, **kwargs):
        return _MockResponse(content)

    return _m


class TestRetrievalMetrics:
    def test_recall(self):
        score, _ = _retrieval_recall(["a", "b", "c"], ["a", "c", "x"], k=None)
        assert score == pytest.approx(2 / 3)

    def test_recall_with_k(self):
        score, _ = _retrieval_recall(["a", "b", "c"], ["c"], k=2)
        assert score == 0.0  # c is at rank 3, outside top-2

    def test_recall_no_gold_is_vacuous(self):
        score, reason = _retrieval_recall(["a"], [], k=None)
        assert score == 1.0
        assert "vacuous" in reason

    def test_precision(self):
        score, _ = _retrieval_precision(["a", "b"], ["a", "x", "y"], k=None)
        assert score == 0.5

    def test_precision_no_retrieval(self):
        score, reason = _retrieval_precision([], ["a"], k=None)
        assert score == 0.0
        assert "no documents" in reason


class TestParseScore:
    def test_json(self):
        assert _parse_score('{"score": 0.8, "reasoning": "ok"}') == (0.8, "ok")

    def test_code_fence(self):
        score, _ = _parse_score('```json\n{"score": 0.5}\n```')
        assert score == 0.5

    def test_bare_number(self):
        assert _parse_score("0.42") == (0.42, None)

    def test_clamped(self):
        assert _parse_score('{"score": 1.7}')[0] == 1.0

    def test_unparseable(self):
        score, reason = _parse_score("not a score")
        assert score == 0.0
        assert "unparseable" in reason


async def test_retrieval_only_judge_is_deterministic():
    judge = RagJudge(criteria=[
        {"name": "retrieval_recall", "type": "retrieval_recall", "k": 3},
        {"name": "retrieval_precision", "type": "retrieval_precision", "k": 3},
    ])
    examples = [EvalExample(input="q", expected="", extra={"relevant_ids": ["d1", "d2"]})]
    results = [TargetResult(
        output="answer", latency_ms=1.0,
        metadata={"retrieved_ids": ["d1", "d9", "d2"]},
    )]

    per_example = await judge.evaluate_dataset(examples, results)
    sub = per_example[0].sub_scores
    assert sub["retrieval_recall"] == 1.0          # both gold ids in top-3
    assert sub["retrieval_precision"] == pytest.approx(2 / 3)  # 2 of 3 retrieved relevant


async def test_llm_criteria_use_metadata_contexts():
    judge = RagJudge(criteria=[{"name": "faithfulness", "type": "faithfulness"}])
    examples = [EvalExample(input="q", expected="")]
    results = [TargetResult(
        output="the sky is blue", latency_ms=1.0,
        metadata={"contexts": ["the sky is blue due to scattering"]},
    )]

    with patch("llmci.judges.rag.litellm.acompletion",
               side_effect=_mock_llm('{"score": 0.9, "reasoning": "supported"}')):
        per_example = await judge.evaluate_dataset(examples, results)

    assert per_example[0].sub_scores["faithfulness"] == 0.9


async def test_faithfulness_without_contexts_scores_zero():
    judge = RagJudge(criteria=[{"name": "faithfulness", "type": "faithfulness"}])
    examples = [EvalExample(input="q", expected="")]
    results = [TargetResult(output="answer", latency_ms=1.0, metadata={})]

    per_example = await judge.evaluate_dataset(examples, results)
    assert per_example[0].sub_scores["faithfulness"] == 0.0
    assert "no contexts" in per_example[0].reason


def test_subscores_become_aggregate_metrics():
    examples = [EvalExample(input="q", expected="") for _ in range(2)]
    results = [TargetResult(output="a", latency_ms=1.0) for _ in range(2)]
    per_example = [
        JudgeResult(score=0.8, sub_scores={"faithfulness": 1.0, "answer_relevance": 0.6}),
        JudgeResult(score=0.6, sub_scores={"faithfulness": 0.5, "answer_relevance": 0.7}),
    ]
    metrics = compute_metrics(
        examples, results, per_example, ["faithfulness", "answer_relevance"]
    )
    assert metrics["faithfulness"] == 0.75
    assert metrics["answer_relevance"] == pytest.approx(0.65)


def test_factory_builds_rag_judge():
    judge = create_judge(JudgeConfig(
        type="rag",
        criteria=[{"name": "retrieval_recall", "type": "retrieval_recall"}],
    ))
    assert isinstance(judge, RagJudge)


def test_factory_rejects_unknown_criterion():
    from llmci.errors import ConfigError

    with pytest.raises(ConfigError):
        create_judge(JudgeConfig(type="rag", criteria=[{"name": "bogus"}]))
