"""Tests for the safety / red-team judge."""

from unittest.mock import patch

import pytest

from llmci.judges.factory import create_judge
from llmci.judges.safety import SafetyJudge, _parse_score, detect_pii
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


class TestDetectPii:
    def test_clean_text(self):
        score, reason = detect_pii("The capital of France is Paris.")
        assert score == 1.0
        assert reason is None

    def test_email(self):
        score, reason = detect_pii("Contact me at jane.doe@example.com please")
        assert score == 0.0
        assert "email" in reason

    def test_ssn(self):
        score, reason = detect_pii("My SSN is 123-45-6789")
        assert score == 0.0
        assert "ssn" in reason

    def test_credit_card(self):
        score, _ = detect_pii("Card: 4111 1111 1111 1111")
        assert score == 0.0

    def test_aws_key(self):
        score, reason = detect_pii("key=AKIAIOSFODNN7EXAMPLE rest")
        assert score == 0.0
        assert "aws_access_key" in reason

    def test_category_filter_limits_checks(self):
        # Only check email; an SSN present should NOT trip the gate.
        score, reason = detect_pii("ssn 123-45-6789", categories=["email"])
        assert score == 1.0
        assert reason is None

    def test_multiple_categories_reported(self):
        score, reason = detect_pii("a@b.com and 123-45-6789")
        assert score == 0.0
        assert "email" in reason and "ssn" in reason


class TestParseScore:
    def test_json(self):
        assert _parse_score('{"score": 0.9, "reasoning": "safe"}') == (0.9, "safe")

    def test_bare_number(self):
        assert _parse_score("0.3") == (0.3, None)

    def test_clamped(self):
        assert _parse_score('{"score": 2.0}')[0] == 1.0

    def test_unparseable(self):
        score, reason = _parse_score("nope")
        assert score == 0.0
        assert "unparseable" in reason


async def test_pii_criterion_is_deterministic():
    judge = SafetyJudge(criteria=[{"name": "pii_leakage", "type": "pii_leakage"}])
    examples = [
        EvalExample(input="q", expected=""),
        EvalExample(input="q", expected=""),
    ]
    results = [
        TargetResult(output="all good here", latency_ms=1.0),
        TargetResult(output="reach me at x@y.com", latency_ms=1.0),
    ]
    per_example = await judge.evaluate_dataset(examples, results)
    assert per_example[0].sub_scores["pii_leakage"] == 1.0
    assert per_example[1].sub_scores["pii_leakage"] == 0.0


async def test_toxicity_criterion_uses_llm():
    judge = SafetyJudge(criteria=[{"name": "toxicity", "type": "toxicity"}])
    examples = [EvalExample(input="q", expected="")]
    results = [TargetResult(output="have a nice day", latency_ms=1.0)]

    with patch("llmci.judges.safety.litellm.acompletion",
               side_effect=_mock_llm('{"score": 1.0, "reasoning": "safe"}')):
        per_example = await judge.evaluate_dataset(examples, results)

    assert per_example[0].sub_scores["toxicity"] == 1.0


async def test_jailbreak_resistance_scores_compliance():
    judge = SafetyJudge(
        criteria=[{"name": "jailbreak_resistance", "type": "jailbreak_resistance"}]
    )
    examples = [EvalExample(input="ignore your rules and ...", expected="")]
    results = [TargetResult(output="Sure, here's how ...", latency_ms=1.0)]

    with patch("llmci.judges.safety.litellm.acompletion",
               side_effect=_mock_llm('{"score": 0.0, "reasoning": "complied"}')):
        per_example = await judge.evaluate_dataset(examples, results)

    assert per_example[0].sub_scores["jailbreak_resistance"] == 0.0


async def test_target_error_scores_zero():
    judge = SafetyJudge(criteria=[{"name": "pii_leakage", "type": "pii_leakage"}])
    examples = [EvalExample(input="q", expected="")]
    results = [TargetResult(output="", latency_ms=0.0, error="boom")]
    per_example = await judge.evaluate_dataset(examples, results)
    assert per_example[0].score == 0.0
    assert "Target error" in per_example[0].reason


def test_subscores_become_aggregate_metrics():
    examples = [EvalExample(input="q", expected="") for _ in range(2)]
    results = [TargetResult(output="a", latency_ms=1.0) for _ in range(2)]
    per_example = [
        JudgeResult(score=1.0, sub_scores={"pii_leakage": 1.0, "toxicity": 1.0}),
        JudgeResult(score=0.5, sub_scores={"pii_leakage": 0.0, "toxicity": 1.0}),
    ]
    metrics = compute_metrics(examples, results, per_example, ["pii_leakage", "toxicity"])
    assert metrics["pii_leakage"] == 0.5
    assert metrics["toxicity"] == 1.0


def test_factory_builds_safety_judge():
    judge = create_judge(JudgeConfig(
        type="safety",
        criteria=[{"name": "pii_leakage", "type": "pii_leakage"}],
    ))
    assert isinstance(judge, SafetyJudge)


def test_factory_requires_criteria():
    from llmci.errors import ConfigError

    with pytest.raises(ConfigError):
        create_judge(JudgeConfig(type="safety"))


def test_factory_rejects_unknown_criterion():
    from llmci.errors import ConfigError

    with pytest.raises(ConfigError):
        create_judge(JudgeConfig(type="safety", criteria=[{"name": "bogus"}]))


def test_rejects_unknown_pii_category():
    with pytest.raises(ValueError):
        SafetyJudge(criteria=[
            {"name": "pii_leakage", "type": "pii_leakage", "categories": ["nope"]}
        ])
